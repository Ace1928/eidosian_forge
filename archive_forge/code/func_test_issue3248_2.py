import copy
import pickle
import numpy
import pytest
from spacy.attrs import DEP, HEAD
from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(3248)
def test_issue3248_2():
    """Test that the PhraseMatcher can be pickled correctly."""
    nlp = English()
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('TEST1', [nlp('a'), nlp('b'), nlp('c')])
    matcher.add('TEST2', [nlp('d')])
    data = pickle.dumps(matcher)
    new_matcher = pickle.loads(data)
    assert len(new_matcher) == len(matcher)