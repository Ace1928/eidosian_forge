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
def test_entity_ruler_remove_several_patterns(nlp, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    patterns = [{'label': 'PERSON', 'pattern': 'Dina', 'id': 'dina'}, {'label': 'ORG', 'pattern': 'ACME', 'id': 'acme'}, {'label': 'ORG', 'pattern': 'ACM'}]
    ruler.add_patterns(patterns)
    doc = nlp('Dina founded her company ACME.')
    assert len(ruler.patterns) == 3
    assert len(doc.ents) == 2
    assert doc.ents[0].label_ == 'PERSON'
    assert doc.ents[0].text == 'Dina'
    assert doc.ents[1].label_ == 'ORG'
    assert doc.ents[1].text == 'ACME'
    if isinstance(ruler, EntityRuler):
        ruler.remove('dina')
    else:
        ruler.remove_by_id('dina')
    doc = nlp('Dina founded her company ACME')
    assert len(ruler.patterns) == 2
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == 'ORG'
    assert doc.ents[0].text == 'ACME'
    if isinstance(ruler, EntityRuler):
        ruler.remove('acme')
    else:
        ruler.remove_by_id('acme')
    doc = nlp('Dina founded her company ACME')
    assert len(ruler.patterns) == 1
    assert len(doc.ents) == 0