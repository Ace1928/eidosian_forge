import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def generate_expanded_graph(graph_in):
    """Generates an expanded graph based on node parameterization

    Parameterization is controlled using the `iterables` field of the
    pipeline elements.  Thus if there are two nodes with iterables a=[1,2]
    and b=[3,4] this procedure will generate a graph with sub-graphs
    parameterized as (a=1,b=3), (a=1,b=4), (a=2,b=3) and (a=2,b=4).
    """
    import networkx as nx
    try:
        dfs_preorder = nx.dfs_preorder
    except AttributeError:
        dfs_preorder = nx.dfs_preorder_nodes
    logger.debug('PE: expanding iterables')
    graph_in = _remove_nonjoin_identity_nodes(graph_in, keep_iterables=True)
    for node in graph_in.nodes():
        if node.iterables:
            _standardize_iterables(node)
    allprefixes = list('abcdefghijklmnopqrstuvwxyz')
    inodes = _iterable_nodes(graph_in)
    logger.debug('Detected iterable nodes %s', inodes)
    while inodes:
        inode = inodes[0]
        logger.debug('Expanding the iterable node %s...', inode)
        jnodes = [node for node in graph_in.nodes() if hasattr(node, 'joinsource') and inode.name == node.joinsource and nx.has_path(graph_in, inode, node)]
        jedge_dict = {}
        for jnode in jnodes:
            in_edges = jedge_dict[jnode] = {}
            edges2remove = []
            for src, dest, data in graph_in.in_edges(jnode, True):
                in_edges[src.itername] = data
                edges2remove.append((src, dest))
            for src, dest in edges2remove:
                graph_in.remove_edge(src, dest)
                logger.debug('Excised the %s -> %s join node in-edge.', src, dest)
        if inode.itersource:
            src_name, src_fields = inode.itersource
            if isinstance(src_fields, (str, bytes)):
                src_fields = [src_fields]
            try:
                iter_src = next((node for node in graph_in.nodes() if node.name == src_name and nx.has_path(graph_in, node, inode)))
            except StopIteration:
                raise ValueError('The node %s itersource %s was not found among the iterable predecessor nodes' % (inode, src_name))
            logger.debug('The node %s has iterable source node %s', inode, iter_src)
            iterables = {}
            src_values = [getattr(iter_src.inputs, field) for field in src_fields]
            if len(src_values) == 1:
                key = src_values[0]
            else:
                key = tuple(src_values)
            iter_dict = dict([(field, lookup[key]) for field, lookup in inode.iterables if key in lookup])

            def make_field_func(*pair):
                return (pair[0], lambda: pair[1])
            iterables = dict([make_field_func(*pair) for pair in list(iter_dict.items())])
        else:
            iterables = inode.iterables.copy()
        inode.iterables = None
        logger.debug('node: %s iterables: %s', inode, iterables)
        subnodes = [s for s in dfs_preorder(graph_in, inode)]
        prior_prefix = [re.findall('\\.(.)I', s._id) for s in subnodes if s._id]
        prior_prefix = sorted([l for item in prior_prefix for l in item])
        if not prior_prefix:
            iterable_prefix = 'a'
        else:
            if prior_prefix[-1] == 'z':
                raise ValueError('Too many iterables in the workflow')
            iterable_prefix = allprefixes[allprefixes.index(prior_prefix[-1]) + 1]
        logger.debug(('subnodes:', subnodes))
        inode._id += '.%sI' % iterable_prefix
        subgraph = graph_in.subgraph(subnodes).copy()
        graph_in = _merge_graphs(graph_in, subnodes, subgraph, inode._hierarchy + inode._id, iterables, iterable_prefix, inode.synchronize)
        for jnode in jnodes:
            old_edge_dict = jedge_dict[jnode]
            expansions = defaultdict(list)
            for node in graph_in:
                for src_id in list(old_edge_dict.keys()):
                    itername = node.itername
                    if hasattr(node, 'joinfield') and itername == src_id:
                        continue
                    if itername.startswith(src_id):
                        suffix = itername[len(src_id):]
                        if re.fullmatch('((\\.[a-z](I\\.[a-z])?|J)\\d+)?', suffix):
                            expansions[src_id].append(node)
            for in_id, in_nodes in list(expansions.items()):
                logger.debug('The join node %s input %s was expanded to %d nodes.', jnode, in_id, len(in_nodes))
            for in_nodes in list(expansions.values()):
                in_nodes.sort(key=lambda node: node._id)
            iter_cnt = count_iterables(iterables, inode.synchronize)
            slot_dicts = [jnode._add_join_item_fields() for _ in range(iter_cnt)]
            for old_id, in_nodes in list(expansions.items()):
                for in_idx, in_node in enumerate(in_nodes):
                    olddata = old_edge_dict[old_id]
                    newdata = deepcopy(olddata)
                    connects = newdata['connect']
                    join_fields = [field for _, field in connects if field in jnode.joinfield]
                    slots = slot_dicts[in_idx]
                    for con_idx, connect in enumerate(connects):
                        src_field, dest_field = connect
                        if dest_field in slots:
                            slot_field = slots[dest_field]
                            connects[con_idx] = (src_field, slot_field)
                            logger.debug('Qualified the %s -> %s join field %s as %s.', in_node, jnode, dest_field, slot_field)
                    graph_in.add_edge(in_node, jnode, **newdata)
                    logger.debug('Connected the join node %s subgraph to the expanded join point %s', jnode, in_node)
        inodes = _iterable_nodes(graph_in)
    for node in graph_in.nodes():
        if node.parameterization:
            node.parameterization = [param for _, param in sorted(node.parameterization)]
    logger.debug('PE: expanding iterables ... done')
    return _remove_nonjoin_identity_nodes(graph_in)