from pyparsing import *
def conversionParseAction(s, l, t):
    return opening + t[0] + closing