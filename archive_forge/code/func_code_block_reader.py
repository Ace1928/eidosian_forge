from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
def code_block_reader(self, stream):
    return [CodeBlock(t.info, t.content) for t in self.parser.parse(stream.read()) if t.level == 0 and t.type in ('fence', 'code_block')]