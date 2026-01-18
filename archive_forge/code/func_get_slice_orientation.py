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
def get_slice_orientation(self):
    """Returns the slice orientation label.

        Returns
        -------
        orientation : {'transverse', 'sagittal', 'coronal'}
        """
    lab = self._get_unique_image_prop('slice orientation')
    return slice_orientation_codes.label[lab]