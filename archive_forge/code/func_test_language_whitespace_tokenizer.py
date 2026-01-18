import itertools
import logging
import warnings
from unittest import mock
import pytest
from thinc.api import CupyOps, NumpyOps, get_current_ops
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import find_matching_language, ignore_error, raise_error, registry
from spacy.vocab import Vocab
from .util import add_vecs_to_vocab, assert_docs_equal
def test_language_whitespace_tokenizer():
    """Test the custom whitespace tokenizer from the docs."""

    class WhitespaceTokenizer:

        def __init__(self, vocab):
            self.vocab = vocab

        def __call__(self, text):
            words = text.split(' ')
            spaces = [True] * len(words)
            for i, word in enumerate(words):
                if word == '':
                    words[i] = ' '
                    spaces[i] = False
            if words[-1] == ' ':
                words = words[0:-1]
                spaces = spaces[0:-1]
            else:
                spaces[-1] = False
            return Doc(self.vocab, words=words, spaces=spaces)
    nlp = spacy.blank('en')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    text = "   What's happened to    me? he thought. It wasn't a dream.    "
    doc = nlp(text)
    assert doc.text == text