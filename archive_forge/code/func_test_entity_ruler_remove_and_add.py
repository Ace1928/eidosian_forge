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
def test_entity_ruler_remove_and_add(nlp, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    patterns = [{'label': 'DATE', 'pattern': 'last time'}]
    ruler.add_patterns(patterns)
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers'))
    assert len(ruler.patterns) == 1
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == 'DATE'
    assert doc.ents[0].text == 'last time'
    patterns1 = [{'label': 'DATE', 'pattern': 'this time', 'id': 'ttime'}]
    ruler.add_patterns(patterns1)
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers'))
    assert len(ruler.patterns) == 2
    assert len(doc.ents) == 2
    assert doc.ents[0].label_ == 'DATE'
    assert doc.ents[0].text == 'last time'
    assert doc.ents[1].label_ == 'DATE'
    assert doc.ents[1].text == 'this time'
    if isinstance(ruler, EntityRuler):
        ruler.remove('ttime')
    else:
        ruler.remove_by_id('ttime')
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers'))
    assert len(ruler.patterns) == 1
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == 'DATE'
    assert doc.ents[0].text == 'last time'
    ruler.add_patterns(patterns1)
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers'))
    assert len(ruler.patterns) == 2
    assert len(doc.ents) == 2
    patterns2 = [{'label': 'DATE', 'pattern': 'another time', 'id': 'ttime'}]
    ruler.add_patterns(patterns2)
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers, another time some chocolate.'))
    assert len(ruler.patterns) == 3
    assert len(doc.ents) == 3
    if isinstance(ruler, EntityRuler):
        ruler.remove('ttime')
    else:
        ruler.remove_by_id('ttime')
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers, another time some chocolate.'))
    assert len(ruler.patterns) == 1
    assert len(doc.ents) == 1