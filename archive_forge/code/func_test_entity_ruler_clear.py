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
def test_entity_ruler_clear(nlp, patterns, entity_ruler_factory):
    """Test that initialization clears patterns."""
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    ruler.add_patterns(patterns)
    assert len(ruler.labels) == 4
    doc = nlp('hello world')
    assert len(doc.ents) == 1
    ruler.clear()
    assert len(ruler.labels) == 0
    with pytest.warns(UserWarning):
        doc = nlp('hello world')
    assert len(doc.ents) == 0