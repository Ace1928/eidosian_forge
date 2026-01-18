import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def b_value(self):
    """Return b value for diffusion or None if not available"""
    q_vec = self.q_vector
    if q_vec is None:
        return None
    return q2bg(q_vec)[0]