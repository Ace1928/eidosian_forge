import hypothesis
import hypothesis.strategies
import numpy
import pytest
from thinc.tests.strategies import ndarrays_of_shape
from spacy.language import Language
from spacy.pipeline._parser_internals._beam_utils import BeamBatch
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
@pytest.fixture(scope='module')
def moves(vocab):
    aeager = ArcEager(vocab.strings, {})
    aeager.add_action(0, '')
    aeager.add_action(1, '')
    aeager.add_action(2, 'nsubj')
    aeager.add_action(2, 'punct')
    aeager.add_action(2, 'aux')
    aeager.add_action(2, 'nsubjpass')
    aeager.add_action(3, 'dobj')
    aeager.add_action(2, 'aux')
    aeager.add_action(4, 'ROOT')
    return aeager