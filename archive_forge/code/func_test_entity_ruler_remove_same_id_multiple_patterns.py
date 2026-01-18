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
def test_entity_ruler_remove_same_id_multiple_patterns(nlp, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    patterns = [{'label': 'PERSON', 'pattern': 'Dina', 'id': 'dina'}, {'label': 'ORG', 'pattern': 'DinaCorp', 'id': 'dina'}, {'label': 'ORG', 'pattern': 'ACME', 'id': 'acme'}]
    ruler.add_patterns(patterns)
    doc = nlp('Dina founded DinaCorp and ACME.')
    assert len(ruler.patterns) == 3
    if isinstance(ruler, EntityRuler):
        assert 'PERSON||dina' in ruler.phrase_matcher
        assert 'ORG||dina' in ruler.phrase_matcher
    assert len(doc.ents) == 3
    if isinstance(ruler, EntityRuler):
        ruler.remove('dina')
    else:
        ruler.remove_by_id('dina')
    doc = nlp('Dina founded DinaCorp and ACME.')
    assert len(ruler.patterns) == 1
    if isinstance(ruler, EntityRuler):
        assert 'PERSON||dina' not in ruler.phrase_matcher
        assert 'ORG||dina' not in ruler.phrase_matcher
    assert len(doc.ents) == 1