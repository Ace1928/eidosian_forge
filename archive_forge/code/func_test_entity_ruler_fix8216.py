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
@pytest.mark.issue(8216)
@pytest.mark.parametrize('entity_ruler_factory', ENTITY_RULERS)
def test_entity_ruler_fix8216(nlp, patterns, entity_ruler_factory):
    """Test that patterns don't get added excessively."""
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler', config={'validate': True})
    ruler.add_patterns(patterns)
    pattern_count = sum((len(mm) for mm in ruler.matcher._patterns.values()))
    assert pattern_count > 0
    ruler.add_patterns([])
    after_count = sum((len(mm) for mm in ruler.matcher._patterns.values()))
    assert after_count == pattern_count