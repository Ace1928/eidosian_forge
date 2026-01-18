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
def test_labels_from_BILUO():
    """Test that labels are inferred correctly when there's a - in label."""
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('LARGE-ANIMAL')
    nlp.initialize()
    move_names = ['O', 'B-LARGE-ANIMAL', 'I-LARGE-ANIMAL', 'L-LARGE-ANIMAL', 'U-LARGE-ANIMAL']
    labels = {'LARGE-ANIMAL'}
    assert ner.move_names == move_names
    assert set(ner.labels) == labels