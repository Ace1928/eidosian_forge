import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(6839)
def test_issue6839(en_vocab):
    """Ensure that PhraseMatcher accepts Span as input"""
    words = ['I', 'like', 'Spans', 'and', 'Docs', 'in', 'my', 'input', ',', 'and', 'nothing', 'else', '.']
    doc = Doc(en_vocab, words=words)
    span = doc[:8]
    pattern = Doc(en_vocab, words=['Spans', 'and', 'Docs'])
    matcher = PhraseMatcher(en_vocab)
    matcher.add('SPACY', [pattern])
    matches = matcher(span)
    assert matches