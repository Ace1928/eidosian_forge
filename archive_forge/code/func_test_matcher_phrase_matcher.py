import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_matcher_phrase_matcher(en_vocab):
    doc = Doc(en_vocab, words=['I', 'like', 'Google', 'Now', 'best'])
    pattern = Doc(en_vocab, words=['Google', 'Now'])
    matcher = PhraseMatcher(en_vocab)
    matcher.add('COMPANY', [pattern])
    assert len(matcher(doc)) == 1
    pattern = Doc(en_vocab, words=['I'])
    matcher = PhraseMatcher(en_vocab)
    matcher.add('I', [pattern])
    assert len(matcher(doc)) == 1
    pattern = Doc(en_vocab, words=['I', 'like'])
    matcher = PhraseMatcher(en_vocab)
    matcher.add('ILIKE', [pattern])
    assert len(matcher(doc)) == 1
    pattern = Doc(en_vocab, words=['best'])
    matcher = PhraseMatcher(en_vocab)
    matcher.add('BEST', [pattern])
    assert len(matcher(doc)) == 1
    pattern = Doc(en_vocab, words=['Now', 'best'])
    matcher = PhraseMatcher(en_vocab)
    matcher.add('NOWBEST', [pattern])
    assert len(matcher(doc)) == 1