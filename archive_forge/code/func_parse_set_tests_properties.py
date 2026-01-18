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
def parse_set_tests_properties(self, tokens, _breakstack):
    """Parse a set_tests_properties() statement. This statement can set
       properties (key/value strings) on one or more tests.
    """
    test_names = []
    properties = {}
    while tokens and tokens[0].spelling != 'PROPERTIES' and (tokens[0].type != lex.TokenType.RIGHT_PAREN):
        token = tokens.pop(0)
        if token.type == lex.TokenType.WHITESPACE:
            continue
        assert token.type in (lex.TokenType.WORD, lex.TokenType.UNQUOTED_LITERAL), 'Unexpected {}'.format(token)
        test_names.append(token.spelling)
    token = tokens.pop(0)
    assert token.spelling == 'PROPERTIES'
    while tokens and tokens[0].type != lex.TokenType.RIGHT_PAREN:
        token = tokens.pop(0)
        if token.type is lex.TokenType.WHITESPACE:
            token = tokens.pop(0)
        assert token.type in (lex.TokenType.WORD, lex.TokenType.UNQUOTED_LITERAL), 'Unexpected {}'.format(token)
        key = token.spelling
        token = tokens.pop(0)
        if token.type is lex.TokenType.WHITESPACE:
            token = tokens.pop(0)
        if token.type is lex.TokenType.QUOTED_LITERAL:
            value = token.spelling[1:-1]
        else:
            value = token.spelling
        properties[key] = value
    for test_name in test_names:
        logger.debug('Updating properties for %s', test_name)
        self.tests[test_name].props.update(properties)