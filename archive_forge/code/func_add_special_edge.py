import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def add_special_edge(self, src, dst, srcport='', dstport='', **kwds):
    src_is_graph = isinstance(src, DotSubGraph)
    dst_is_graph = isinstance(dst, DotSubGraph)
    edges = []
    if src_is_graph:
        src_nodes = src.get_all_nodes()
    else:
        src_nodes = [src]
    if dst_is_graph:
        dst_nodes = dst.get_all_nodes()
    else:
        dst_nodes = [dst]
    for src_node in src_nodes:
        for dst_node in dst_nodes:
            edge = self.add_edge(src_node, dst_node, srcport, dstport, **kwds)
            edges.append(edge)
    return edges