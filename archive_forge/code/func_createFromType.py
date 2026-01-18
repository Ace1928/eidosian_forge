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
def createFromType(self, tokenType, text):
    assert isinstance(tokenType, six.integer_types), type(tokenType).__name__
    assert isinstance(text, six.string_types), type(text).__name__
    fromToken = self.createToken(tokenType=tokenType, text=text)
    t = self.createWithPayload(fromToken)
    return t