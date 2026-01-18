from __future__ import with_statement
import logging
import os
import random
import re
import sys
from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import (
from gensim.utils import deaccent, simple_tokenize
from smart_open import open
def iter_filepaths(self):
    """Generate (lazily)  paths to each file in the directory structure within the specified range of depths.
        If a filename pattern to match was given, further filter to only those filenames that match.

        Yields
        ------
        str
            Path to file

        """
    for depth, dirpath, dirnames, filenames in walk(self.input):
        if self.min_depth <= depth <= self.max_depth:
            if self.pattern is not None:
                filenames = (n for n in filenames if self.pattern.match(n) is not None)
            if self.exclude_pattern is not None:
                filenames = (n for n in filenames if self.exclude_pattern.match(n) is None)
            for name in filenames:
                yield os.path.join(dirpath, name)