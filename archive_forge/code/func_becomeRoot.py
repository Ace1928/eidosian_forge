from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import UP, DOWN, EOF, INVALID_TOKEN_TYPE
from antlr3.exceptions import MismatchedTreeNodeException, \
from antlr3.recognizers import BaseRecognizer, RuleReturnScope
from antlr3.streams import IntStream
from antlr3.tokens import CommonToken, Token, INVALID_TOKEN
import six
from six.moves import range
def becomeRoot(self, newRoot, oldRoot):
    """
        If oldRoot is a nil root, just copy or move the children to newRoot.
        If not a nil root, make oldRoot a child of newRoot.

          old=^(nil a b c), new=r yields ^(r a b c)
          old=^(a b c), new=r yields ^(r ^(a b c))

        If newRoot is a nil-rooted single child tree, use the single
        child as the new root node.

          old=^(nil a b c), new=^(nil r) yields ^(r a b c)
          old=^(a b c), new=^(nil r) yields ^(r ^(a b c))

        If oldRoot was null, it's ok, just return newRoot (even if isNil).

          old=null, new=r yields r
          old=null, new=^(nil r) yields ^(nil r)

        Return newRoot.  Throw an exception if newRoot is not a
        simple node or nil root with a single child node--it must be a root
        node.  If newRoot is ^(nil x) return x as newRoot.

        Be advised that it's ok for newRoot to point at oldRoot's
        children; i.e., you don't have to copy the list.  We are
        constructing these nodes so we should have this control for
        efficiency.
        """
    if isinstance(newRoot, Token):
        newRoot = self.create(newRoot)
    if oldRoot is None:
        return newRoot
    if not isinstance(newRoot, CommonTree):
        newRoot = self.createWithPayload(newRoot)
    if newRoot.isNil():
        nc = newRoot.getChildCount()
        if nc == 1:
            newRoot = newRoot.getChild(0)
        elif nc > 1:
            raise RuntimeError('more than one node as root')
    newRoot.addChild(oldRoot)
    return newRoot