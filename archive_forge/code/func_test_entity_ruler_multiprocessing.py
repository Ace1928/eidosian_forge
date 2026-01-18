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
@pytest.mark.parametrize('n_process', [1, 2])
@pytest.mark.parametrize('entity_ruler_factory', ENTITY_RULERS)
def test_entity_ruler_multiprocessing(nlp, n_process, entity_ruler_factory):
    if isinstance(get_current_ops, NumpyOps) or n_process < 2:
        texts = ['I enjoy eating Pizza Hut pizza.']
        patterns = [{'label': 'FASTFOOD', 'pattern': 'Pizza Hut', 'id': '1234'}]
        ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
        ruler.add_patterns(patterns)
        for doc in nlp.pipe(texts, n_process=2):
            for ent in doc.ents:
                assert ent.ent_id_ == '1234'