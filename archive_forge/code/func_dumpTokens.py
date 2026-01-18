import time
import pprint
import sys
from pyparsing import Literal, Keyword, Word, OneOrMore, ZeroOrMore, \
import pyparsing
def dumpTokens(s, l, t):
    import pprint
    pprint.pprint(t.asList())