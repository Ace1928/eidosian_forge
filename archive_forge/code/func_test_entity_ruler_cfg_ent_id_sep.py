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
def test_entity_ruler_cfg_ent_id_sep(nlp, patterns, entity_ruler_factory):
    config = {'overwrite_ents': True, 'ent_id_sep': '**'}
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler', config=config)
    ruler.add_patterns(patterns)
    doc = nlp('Apple is a technology company')
    if isinstance(ruler, EntityRuler):
        assert 'TECH_ORG**a1' in ruler.phrase_patterns
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == 'TECH_ORG'
    assert doc.ents[0].ent_id_ == 'a1'