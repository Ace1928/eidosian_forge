import pytest
from thinc.api import NumpyOps, get_current_ops
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import EntityRecognizer, EntityRuler, SpanRuler, merge_entities
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tests.util import make_tempdir
from spacy.tokens import Doc, Span
@pytest.mark.issue(8168)
@pytest.mark.parametrize('entity_ruler_factory', ENTITY_RULERS)
def test_issue8168(entity_ruler_factory):
    nlp = English()
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    patterns = [{'label': 'ORG', 'pattern': 'Apple'}, {'label': 'GPE', 'pattern': [{'LOWER': 'san'}, {'LOWER': 'francisco'}], 'id': 'san-francisco'}, {'label': 'GPE', 'pattern': [{'LOWER': 'san'}, {'LOWER': 'fran'}], 'id': 'san-francisco'}]
    ruler.add_patterns(patterns)
    doc = nlp('San Francisco San Fran')
    assert all((t.ent_id_ == 'san-francisco' for t in doc))