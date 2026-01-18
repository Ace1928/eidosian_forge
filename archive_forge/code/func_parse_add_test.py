import argparse
import collections
import io
import json
import logging
import os
import sys
from xml.etree import ElementTree as ET
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang import parse
def parse_add_test(self, tokens, _breakstack):
    """Parse an add_test() statement. This statement contains the test name
       and the command for the test.
    """
    if not tokens:
        raise RuntimeError('Ran out of tokens while processing add_test')
    if tokens[0].type not in (lex.TokenType.WORD, lex.TokenType.UNQUOTED_LITERAL):
        raise RuntimeError('Expected a WORD token but got {}'.format(tokens[0]))
    test_name = tokens[0].spelling
    tokens.pop(0)
    test_argv = []
    while tokens and tokens[0].type != lex.TokenType.RIGHT_PAREN:
        token = tokens.pop(0)
        if token.type is lex.TokenType.WHITESPACE:
            continue
        if token.type is lex.TokenType.QUOTED_LITERAL:
            spelling = token.spelling[1:-1]
        else:
            spelling = token.spelling
        test_argv.append(spelling)
    logger.debug('Adding test %s', test_name)
    self.tests[test_name] = TestSpec(test_name, test_argv, self.cwd)