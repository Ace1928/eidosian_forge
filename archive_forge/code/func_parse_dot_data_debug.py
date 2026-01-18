import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def parse_dot_data_debug(self, data):
    """Parse dot data"""
    try:
        try:
            self.dotparser.parseWithTabs()
        except:
            log.warning('Old version of pyparsing. Parser may not work correctly')
        tokens = self.dotparser.parseString(data)
        self.build_top_graph(tokens[0])
        return tokens[0]
    except ParseException as err:
        print(err.line)
        print(' ' * (err.column - 1) + '^')
        print(err)
        return None