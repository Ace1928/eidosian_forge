import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
def get_q_vectors(self):
    """Get Q vectors from the data

        Returns
        -------
        q_vectors : None or array
            Array of q vectors (bvals * bvecs), or None if not a diffusion
            acquisition.
        """
    bvals, bvecs = self.get_bvals_bvecs()
    if bvals is None or bvecs is None:
        return None
    return bvecs * bvals[:, np.newaxis]