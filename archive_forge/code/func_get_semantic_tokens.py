from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
def get_semantic_tokens(self, out=None):
    """
    Recursively reconstruct a stream of semantic tokens
    """
    return self.get_tokens(out, kind='semantic')