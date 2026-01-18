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
def test_entity_ruler_remove_patterns_in_a_row(nlp, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    patterns = [{'label': 'PERSON', 'pattern': 'Dina', 'id': 'dina'}, {'label': 'ORG', 'pattern': 'ACME', 'id': 'acme'}, {'label': 'DATE', 'pattern': 'her birthday', 'id': 'bday'}, {'label': 'ORG', 'pattern': 'ACM'}]
    ruler.add_patterns(patterns)
    doc = nlp('Dina founded her company ACME on her birthday')
    assert len(doc.ents) == 3
    assert doc.ents[0].label_ == 'PERSON'
    assert doc.ents[0].text == 'Dina'
    assert doc.ents[1].label_ == 'ORG'
    assert doc.ents[1].text == 'ACME'
    assert doc.ents[2].label_ == 'DATE'
    assert doc.ents[2].text == 'her birthday'
    if isinstance(ruler, EntityRuler):
        ruler.remove('dina')
        ruler.remove('acme')
        ruler.remove('bday')
    else:
        ruler.remove_by_id('dina')
        ruler.remove_by_id('acme')
        ruler.remove_by_id('bday')
    doc = nlp('Dina went to school')
    assert len(doc.ents) == 0