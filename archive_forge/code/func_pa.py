import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def pa(s, l, tokens):
    for attrName, attrValue in attrs:
        if attrName not in tokens:
            raise ParseException(s, l, 'no matching attribute ' + attrName)
        if attrValue != withAttribute.ANY_VALUE and tokens[attrName] != attrValue:
            raise ParseException(s, l, "attribute '%s' has value '%s', must be '%s'" % (attrName, tokens[attrName], attrValue))