import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_attr_validation(en_vocab):
    with pytest.raises(ValueError):
        PhraseMatcher(en_vocab, attr='UNSUPPORTED')