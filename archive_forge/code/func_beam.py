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
@pytest.fixture
def beam(moves, examples, beam_width):
    states, golds, _ = moves.init_gold_batch(examples)
    return BeamBatch(moves, states, golds, width=beam_width, density=0.0)