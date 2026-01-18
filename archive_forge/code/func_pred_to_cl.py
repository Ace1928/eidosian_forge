from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor
def pred_to_cl(pred):
    """
        Map a class id to a chord label.
        0..11 major chords, 12..23 minor chords, 24 no chord
        """
    if pred == 24:
        return 'N'
    return '{}:{}'.format(pitch_class_to_label[pred % 12], 'maj' if pred < 12 else 'min')