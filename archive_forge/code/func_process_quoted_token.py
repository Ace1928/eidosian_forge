import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def process_quoted_token(self, data_idx, data):
    token = ''
    c = data[data_idx]
    i = data_idx
    for start, end, escape, incl_quotes in self.quote_chars:
        if c == start:
            if incl_quotes:
                token += c
            i += 1
            while data[i] != end:
                if data[i] == escape:
                    if incl_quotes:
                        token += data[i]
                    i += 1
                    if len(data) == i:
                        raise LogicalExpressionException(None, 'End of input reached.  Escape character [%s] found at end.' % escape)
                    token += data[i]
                else:
                    token += data[i]
                i += 1
                if len(data) == i:
                    raise LogicalExpressionException(None, 'End of input reached.  Expected: [%s]' % end)
            if incl_quotes:
                token += data[i]
            i += 1
            if not token:
                raise LogicalExpressionException(None, 'Empty quoted token found')
            break
    return (token, i)