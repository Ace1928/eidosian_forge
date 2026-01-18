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
@pytest.mark.issue(3209)
def test_issue3209():
    """Test issue that occurred in spaCy nightly where NER labels were being
    mapped to classes incorrectly after loading the model, when the labels
    were added using ner.add_label().
    """
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('ANIMAL')
    nlp.initialize()
    move_names = ['O', 'B-ANIMAL', 'I-ANIMAL', 'L-ANIMAL', 'U-ANIMAL']
    assert ner.move_names == move_names
    nlp2 = English()
    ner2 = nlp2.add_pipe('ner')
    model = ner2.model
    model.attrs['resize_output'](model, ner.moves.n_moves)
    nlp2.from_bytes(nlp.to_bytes())
    assert ner2.move_names == move_names