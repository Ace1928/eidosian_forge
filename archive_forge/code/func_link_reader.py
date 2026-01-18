from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
def link_reader(self, stream):
    return [Link(inline_token.children[i + 1].content, child_token.attrGet('href'), child_token.attrGet('title')) for inline_token in filter(lambda t: t.type == 'inline', self.parser.parse(stream.read())) for i, child_token in enumerate(inline_token.children) if child_token.type == 'link_open']