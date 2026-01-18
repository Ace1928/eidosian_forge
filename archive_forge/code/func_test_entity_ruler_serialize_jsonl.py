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
def test_entity_ruler_serialize_jsonl(nlp, patterns, entity_ruler_factory):
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    ruler.add_patterns(patterns)
    with make_tempdir() as d:
        ruler.to_disk(d / 'test_ruler.jsonl')
        ruler.from_disk(d / 'test_ruler.jsonl')
        with pytest.raises(ValueError):
            ruler.from_disk(d / 'non_existing.jsonl')