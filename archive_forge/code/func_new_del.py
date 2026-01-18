from *Random walks for image segmentation*, Leo Grady, IEEE Trans
import numpy as np
from scipy import sparse, ndimage as ndi
from .._shared import utils
from .._shared.utils import warn
from .._shared.compat import SCIPY_CG_TOL_PARAM_NAME
from ..util import img_as_float
from scipy.sparse.linalg import cg, spsolve
def new_del(self):
    try:
        old_del(self)
    except AttributeError:
        pass