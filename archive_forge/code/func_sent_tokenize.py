import re
from nltk.data import load
from nltk.tokenize.casual import TweetTokenizer, casual_tokenize
from nltk.tokenize.destructive import NLTKWordTokenizer
from nltk.tokenize.legality_principle import LegalitySyllableTokenizer
from nltk.tokenize.mwe import MWETokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.regexp import (
from nltk.tokenize.repp import ReppTokenizer
from nltk.tokenize.sexpr import SExprTokenizer, sexpr_tokenize
from nltk.tokenize.simple import (
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.tokenize.util import regexp_span_tokenize, string_span_tokenize
def sent_tokenize(text, language='english'):
    """
    Return a sentence-tokenized copy of *text*,
    using NLTK's recommended sentence tokenizer
    (currently :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    tokenizer = load(f'tokenizers/punkt/{language}.pickle')
    return tokenizer.tokenize(text)