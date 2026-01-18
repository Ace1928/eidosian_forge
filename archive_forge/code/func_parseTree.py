from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
def parseTree(self):
    if self.ttype != BEGIN:
        return None
    self.ttype = self.tokenizer.nextToken()
    root = self.parseNode()
    if root is None:
        return None
    while self.ttype in (BEGIN, ID, PERCENT, DOT):
        if self.ttype == BEGIN:
            subtree = self.parseTree()
            self.adaptor.addChild(root, subtree)
        else:
            child = self.parseNode()
            if child is None:
                return None
            self.adaptor.addChild(root, child)
    if self.ttype != END:
        return None
    self.ttype = self.tokenizer.nextToken()
    return root