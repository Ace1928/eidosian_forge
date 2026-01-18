import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def fix_keys_for_gexf(orig):
    """
    GEXF Networks can be read in Gephi, however, the keys for the node and edge IDs must be converted to strings
    """
    import networkx as nx
    ntwk = nx.Graph()
    nodes = list(orig.nodes())
    edges = list(orig.edges())
    for node in nodes:
        newnodedata = {}
        newnodedata.update(orig.nodes[node])
        if 'dn_fsname' in orig.nodes[node]:
            newnodedata['label'] = orig.nodes[node]['dn_fsname']
        ntwk.add_node(str(node), **newnodedata)
        if 'dn_position' in ntwk.nodes[str(node)] and 'dn_position' in newnodedata:
            ntwk.nodes[str(node)]['dn_position'] = str(newnodedata['dn_position'])
    for edge in edges:
        data = {}
        data = orig.edge[edge[0]][edge[1]]
        ntwk.add_edge(str(edge[0]), str(edge[1]), **data)
        if 'fiber_length_mean' in ntwk.edge[str(edge[0])][str(edge[1])]:
            ntwk.edge[str(edge[0])][str(edge[1])]['fiber_length_mean'] = str(data['fiber_length_mean'])
        if 'fiber_length_std' in ntwk.edge[str(edge[0])][str(edge[1])]:
            ntwk.edge[str(edge[0])][str(edge[1])]['fiber_length_std'] = str(data['fiber_length_std'])
        if 'number_of_fibers' in ntwk.edge[str(edge[0])][str(edge[1])]:
            ntwk.edge[str(edge[0])][str(edge[1])]['number_of_fibers'] = str(data['number_of_fibers'])
        if 'value' in ntwk.edge[str(edge[0])][str(edge[1])]:
            ntwk.edge[str(edge[0])][str(edge[1])]['value'] = str(data['value'])
    return ntwk