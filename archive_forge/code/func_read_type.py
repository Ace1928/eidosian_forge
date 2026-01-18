import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def read_type(type_string):
    assert isinstance(type_string, str)
    type_string = type_string.replace(' ', '')
    if type_string[0] == '<':
        assert type_string[-1] == '>'
        paren_count = 0
        for i, char in enumerate(type_string):
            if char == '<':
                paren_count += 1
            elif char == '>':
                paren_count -= 1
                assert paren_count > 0
            elif char == ',':
                if paren_count == 1:
                    break
        return ComplexType(read_type(type_string[1:i]), read_type(type_string[i + 1:-1]))
    elif type_string[0] == '%s' % ENTITY_TYPE:
        return ENTITY_TYPE
    elif type_string[0] == '%s' % TRUTH_TYPE:
        return TRUTH_TYPE
    elif type_string[0] == '%s' % ANY_TYPE:
        return ANY_TYPE
    else:
        raise LogicalExpressionException(None, "Unexpected character: '%s'." % type_string[0])