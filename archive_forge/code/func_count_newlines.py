from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
def count_newlines(self):
    newline_count = 0
    for child in self.children:
        newline_count += child.count_newlines()
    return newline_count