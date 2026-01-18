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
@pytest.mark.issue(3830)
def test_issue3830_with_subtok():
    """Test that the parser does have subtok label if learn_tokens=True."""
    config = {'learn_tokens': True}
    model = registry.resolve({'model': DEFAULT_PARSER_MODEL}, validate=True)['model']
    parser = DependencyParser(Vocab(), model, **config)
    parser.add_label('nsubj')
    assert 'subtok' not in parser.labels
    parser.initialize(lambda: [_parser_example(parser)])
    assert 'subtok' in parser.labels