import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def dot_digraph(self):
    s = 'digraph nltk_chart {\n'
    s += '  rankdir=LR;\n'
    s += '  node [height=0.1,width=0.1];\n'
    s += '  node [style=filled, color="lightgray"];\n'
    for y in range(self.num_edges(), -1, -1):
        if y == 0:
            s += '  node [style=filled, color="black"];\n'
        for x in range(self.num_leaves() + 1):
            if y == 0 or (x <= self._edges[y - 1].start() or x >= self._edges[y - 1].end()):
                s += '  %04d.%04d [label=""];\n' % (x, y)
    s += '  x [style=invis]; x->0000.0000 [style=invis];\n'
    for x in range(self.num_leaves() + 1):
        s += '  {rank=same;'
        for y in range(self.num_edges() + 1):
            if y == 0 or (x <= self._edges[y - 1].start() or x >= self._edges[y - 1].end()):
                s += ' %04d.%04d' % (x, y)
        s += '}\n'
    s += '  edge [style=invis, weight=100];\n'
    s += '  node [shape=plaintext]\n'
    s += '  0000.0000'
    for x in range(self.num_leaves()):
        s += '->%s->%04d.0000' % (self.leaf(x), x + 1)
    s += ';\n\n'
    s += '  edge [style=solid, weight=1];\n'
    for y, edge in enumerate(self):
        for x in range(edge.start()):
            s += '  %04d.%04d -> %04d.%04d [style="invis"];\n' % (x, y + 1, x + 1, y + 1)
        s += '  %04d.%04d -> %04d.%04d [label="%s"];\n' % (edge.start(), y + 1, edge.end(), y + 1, edge)
        for x in range(edge.end(), self.num_leaves()):
            s += '  %04d.%04d -> %04d.%04d [style="invis"];\n' % (x, y + 1, x + 1, y + 1)
    s += '}\n'
    return s