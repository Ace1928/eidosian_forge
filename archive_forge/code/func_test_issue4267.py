import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(4267)
def test_issue4267():
    """Test that running an entity_ruler after ner gives consistent results"""
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('PEOPLE')
    nlp.initialize()
    assert 'ner' in nlp.pipe_names
    doc1 = nlp('hi')
    assert doc1.has_annotation('ENT_IOB')
    for token in doc1:
        assert token.ent_iob == 2
    patterns = [{'label': 'SOFTWARE', 'pattern': 'spacy'}]
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    assert 'entity_ruler' in nlp.pipe_names
    assert 'ner' in nlp.pipe_names
    doc2 = nlp('hi')
    assert doc2.has_annotation('ENT_IOB')
    for token in doc2:
        assert token.ent_iob == 2