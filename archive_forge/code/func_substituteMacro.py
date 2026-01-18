from pyparsing import Word, alphas, alphanums, Literal, restOfLine, OneOrMore, \
from pyparsing import dblQuotedString, LineStart
def substituteMacro(s, l, t):
    if t[0] in macros:
        return macros[t[0]]