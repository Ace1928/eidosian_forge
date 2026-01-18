from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
def computeTokenTypes(tokenNames):
    """
    Compute a dict that is an inverted index of
    tokenNames (which maps int token types to names).
    """
    if tokenNames is None:
        return {}
    return dict(((name, type) for type, name in enumerate(tokenNames)))