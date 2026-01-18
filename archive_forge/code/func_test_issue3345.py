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
@pytest.mark.issue(3345)
@pytest.mark.parametrize('entity_ruler_factory', ENTITY_RULERS)
def test_issue3345(entity_ruler_factory):
    """Test case where preset entity crosses sentence boundary."""
    nlp = English()
    doc = Doc(nlp.vocab, words=['I', 'live', 'in', 'New', 'York'])
    doc[4].is_sent_start = True
    ruler = nlp.add_pipe(entity_ruler_factory, name='entity_ruler')
    ruler.add_patterns([{'label': 'GPE', 'pattern': 'New York'}])
    cfg = {'model': DEFAULT_NER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    ner = EntityRecognizer(doc.vocab, model)
    ner.moves.add_action(5, '')
    ner.add_label('GPE')
    doc = ruler(doc)
    state = ner.moves.init_batch([doc])[0]
    ner.moves.apply_transition(state, 'O')
    ner.moves.apply_transition(state, 'O')
    ner.moves.apply_transition(state, 'O')
    assert ner.moves.is_valid(state, 'B-GPE')