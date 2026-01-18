from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
def getTokenType(self, tokenName):
    """Using the map of token names to token types, return the type."""
    try:
        return self.tokenNameToTypeMap[tokenName]
    except KeyError:
        return INVALID_TOKEN_TYPE