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
@pytest.mark.parametrize('entity_ruler_factory', ENTITY_RULERS)
def test_entity_ruler_fuzzy(nlp, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    patterns = [{'label': 'HELLO', 'pattern': [{'LOWER': {'FUZZY': 'hello'}}]}]
    ruler.add_patterns(patterns)
    doc = nlp('helloo')
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == 'HELLO'