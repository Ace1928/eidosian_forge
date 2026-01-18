import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
@pytest.mark.parametrize('text,words', [("A'B C", ['A', "'", 'B', 'C']), ('A-B', ['A-B'])])
def test_gold_misaligned(en_tokenizer, text, words):
    doc = en_tokenizer(text)
    Example.from_dict(doc, {'words': words})