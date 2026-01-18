import pytest
from numpy.testing import assert_equal
from thinc.api import Adam
from spacy import registry, util
from spacy.attrs import DEP, NORM
from spacy.lang.en import English
from spacy.pipeline import DependencyParser
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
from ..util import apply_transition_sequence, make_tempdir
@pytest.mark.issue(2772)
def test_issue2772(en_vocab):
    """Test that deprojectivization doesn't mess up sentence boundaries."""
    words = ['When', 'we', 'write', 'or', 'communicate', 'virtually', ',', 'we', 'can', 'hide', 'our', 'true', 'feelings', '.']
    heads = [4, 2, 9, 2, 2, 4, 9, 9, 9, 9, 12, 12, 9, 9]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert doc[1].is_sent_start is False