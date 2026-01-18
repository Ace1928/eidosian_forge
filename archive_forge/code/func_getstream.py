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
def getstream(self):
    """Generate documents from the underlying plain text collection (of one or more files).

        Yields
        ------
        str
            One document (if lines_are_documents - True), otherwise - each file is one document.

        """
    num_texts = 0
    for path in self.iter_filepaths():
        with open(path, 'rt', encoding=self.encoding) as f:
            if self.lines_are_documents:
                for line in f:
                    yield line.strip()
                    num_texts += 1
            else:
                yield f.read().strip()
                num_texts += 1
    self.length = num_texts