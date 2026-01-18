from pyparsing import *
def romanNumeralLiteral(numeralString, value):
    return Literal(numeralString).setParseAction(replaceWith(value))