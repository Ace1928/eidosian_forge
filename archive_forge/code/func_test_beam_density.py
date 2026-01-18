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
@hypothesis.given(hyp=hypothesis.strategies.data())
def test_beam_density(moves, examples, beam_width, hyp):
    beam_density = float(hyp.draw(hypothesis.strategies.floats(0.0, 1.0, width=32)))
    states, golds, _ = moves.init_gold_batch(examples)
    beam = BeamBatch(moves, states, golds, width=beam_width, density=beam_density)
    n_state = sum((len(beam) for beam in beam))
    scores = hyp.draw(ndarrays_of_shape((n_state, moves.n_moves)))
    beam.advance(scores)
    for b in beam:
        beam_probs = b.probs
        assert b.min_density == beam_density
        assert beam_probs[-1] >= beam_probs[0] * beam_density