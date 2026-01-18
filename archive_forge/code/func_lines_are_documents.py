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
@lines_are_documents.setter
def lines_are_documents(self, lines_are_documents):
    self._lines_are_documents = lines_are_documents
    self.length = None