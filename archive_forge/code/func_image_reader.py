from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
def image_reader(self, stream):
    return [Image(child_token.content, child_token.attrGet('src'), child_token.attrGet('title')) for inline_token in filter(lambda t: t.type == 'inline', self.parser.parse(stream.read())) for child_token in inline_token.children if child_token.type == 'image']