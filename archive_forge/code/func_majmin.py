import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
@property
def majmin(self):
    return np.mean([e.majmin for e in self.eval_objects])