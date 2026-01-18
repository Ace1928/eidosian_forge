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
def test_entity_ruler_existing_complex(nlp, patterns, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler', config={'overwrite_ents': True})
    ruler.add_patterns(patterns)
    nlp.add_pipe('add_ent', before='entity_ruler')
    doc = nlp('foo foo bye bye')
    assert len(doc.ents) == 2
    assert doc.ents[0].label_ == 'COMPLEX'
    assert doc.ents[1].label_ == 'BYE'
    assert len(doc.ents[0]) == 2
    assert len(doc.ents[1]) == 2