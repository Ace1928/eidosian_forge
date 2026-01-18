import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def reset_workspace(self):
    c = self._workspace.canvas()
    fontsize = int(self._size.get())
    node_font = ('helvetica', -(fontsize + 4), 'bold')
    leaf_font = ('helvetica', -(fontsize + 2))
    if self._tree is not None:
        self._workspace.remove_widget(self._tree)
    start = self._grammar.start().symbol()
    rootnode = TextWidget(c, start, font=node_font, draggable=1)
    leaves = []
    for word in self._text:
        leaves.append(TextWidget(c, word, font=leaf_font, draggable=1))
    self._tree = TreeSegmentWidget(c, rootnode, leaves, color='white')
    self._workspace.add_widget(self._tree)
    for leaf in leaves:
        leaf.move(0, 100)