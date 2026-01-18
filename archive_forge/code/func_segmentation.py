import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
@property
def segmentation(self):
    return np.mean([e.segmentation for e in self.eval_objects])