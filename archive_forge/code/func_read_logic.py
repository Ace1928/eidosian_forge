import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def read_logic(s, logic_parser=None, encoding=None):
    """
    Convert a file of First Order Formulas into a list of {Expression}s.

    :param s: the contents of the file
    :type s: str
    :param logic_parser: The parser to be used to parse the logical expression
    :type logic_parser: LogicParser
    :param encoding: the encoding of the input string, if it is binary
    :type encoding: str
    :return: a list of parsed formulas.
    :rtype: list(Expression)
    """
    if encoding is not None:
        s = s.decode(encoding)
    if logic_parser is None:
        logic_parser = LogicParser()
    statements = []
    for linenum, line in enumerate(s.splitlines()):
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        try:
            statements.append(logic_parser.parse(line))
        except LogicalExpressionException as e:
            raise ValueError(f'Unable to parse line {linenum}: {line}') from e
    return statements