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
def test_entity_ruler_properties(nlp, patterns, entity_ruler_factory):
    ruler = EntityRuler(nlp, patterns=patterns, overwrite_ents=True)
    assert sorted(ruler.labels) == sorted(['HELLO', 'BYE', 'COMPLEX', 'TECH_ORG'])
    assert sorted(ruler.ent_ids) == ['a1', 'a2']