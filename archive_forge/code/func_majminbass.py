import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
@property
def majminbass(self):
    return np.mean([e.majminbass for e in self.eval_objects])