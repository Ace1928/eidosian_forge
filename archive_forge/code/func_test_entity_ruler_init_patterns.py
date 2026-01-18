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
def test_entity_ruler_init_patterns(nlp, patterns, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    assert len(ruler.labels) == 0
    ruler.initialize(lambda: [], patterns=patterns)
    assert len(ruler.labels) == 4
    doc = nlp('hello world bye bye')
    assert doc.ents[0].label_ == 'HELLO'
    assert doc.ents[1].label_ == 'BYE'
    nlp.remove_pipe('entity_ruler')
    nlp.config['initialize']['components']['entity_ruler'] = {'patterns': {'@misc': 'entity_ruler_patterns'}}
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    assert len(ruler.labels) == 0
    nlp.initialize()
    assert len(ruler.labels) == 4
    doc = nlp('hello world bye bye')
    assert doc.ents[0].label_ == 'HELLO'
    assert doc.ents[1].label_ == 'BYE'