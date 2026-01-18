import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(3972)
def test_issue3972(en_vocab):
    """Test that the PhraseMatcher returns duplicates for duplicate match IDs."""
    matcher = PhraseMatcher(en_vocab)
    matcher.add('A', [Doc(en_vocab, words=['New', 'York'])])
    matcher.add('B', [Doc(en_vocab, words=['New', 'York'])])
    doc = Doc(en_vocab, words=['I', 'live', 'in', 'New', 'York'])
    matches = matcher(doc)
    assert len(matches) == 2
    found_ids = [en_vocab.strings[ent_id] for ent_id, _, _ in matches]
    assert 'A' in found_ids
    assert 'B' in found_ids