import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(3331)
def test_issue3331(en_vocab):
    """Test that duplicate patterns for different rules result in multiple
    matches, one per rule.
    """
    matcher = PhraseMatcher(en_vocab)
    matcher.add('A', [Doc(en_vocab, words=['Barack', 'Obama'])])
    matcher.add('B', [Doc(en_vocab, words=['Barack', 'Obama'])])
    doc = Doc(en_vocab, words=['Barack', 'Obama', 'lifts', 'America'])
    matches = matcher(doc)
    assert len(matches) == 2
    match_ids = [en_vocab.strings[matches[0][0]], en_vocab.strings[matches[1][0]]]
    assert sorted(match_ids) == ['A', 'B']