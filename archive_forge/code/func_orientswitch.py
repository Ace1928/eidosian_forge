from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def orientswitch(treewidget):
    if treewidget['orientation'] == 'horizontal':
        treewidget.expanded_tree(1, 1).subtrees()[0].set_text('vertical')
        treewidget.collapsed_tree(1, 1).subtrees()[0].set_text('vertical')
        treewidget.collapsed_tree(1).subtrees()[1].set_text('vertical')
        treewidget.collapsed_tree().subtrees()[3].set_text('vertical')
        treewidget['orientation'] = 'vertical'
    else:
        treewidget.expanded_tree(1, 1).subtrees()[0].set_text('horizontal')
        treewidget.collapsed_tree(1, 1).subtrees()[0].set_text('horizontal')
        treewidget.collapsed_tree(1).subtrees()[1].set_text('horizontal')
        treewidget.collapsed_tree().subtrees()[3].set_text('horizontal')
        treewidget['orientation'] = 'horizontal'