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
def get_volume_labels(self):
    """Dynamic labels corresponding to the final data dimension(s).

        This is useful for custom data sorting.  A subset of the info in
        ``self.image_defs`` is returned in an order that matches the final
        data dimension(s).  Only labels that have more than one unique value
        across the dataset will be returned.

        Returns
        -------
        sort_info : dict
            Each key corresponds to volume labels for a dynamically varying
            sequence dimension.  The ordering of the labels matches the volume
            ordering determined via ``self.get_sorted_slice_indices``.
        """
    sorted_indices = self.get_sorted_slice_indices()
    image_defs = self.image_defs
    dynamic_keys = ['cardiac phase number', 'echo number', 'label type', 'image_type_mr', 'dynamic scan number', 'scanning sequence', 'gradient orientation number', 'diffusion b value number']
    dynamic_keys = [d for d in dynamic_keys if d in image_defs.dtype.fields]
    non_unique_keys = []
    for key in dynamic_keys:
        ndim = image_defs[key].ndim
        if ndim == 1:
            num_unique = len(np.unique(image_defs[key]))
        else:
            raise ValueError('unexpected image_defs shape > 1D')
        if num_unique > 1:
            non_unique_keys.append(key)
    sl1_indices = image_defs['slice number'][sorted_indices] == 1
    sort_info = OrderedDict()
    for key in non_unique_keys:
        sort_info[key] = image_defs[key][sorted_indices][sl1_indices]
    return sort_info