import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def get_patient_orient(self):
    """gets orientation of patient based on code stored
        in header, not always reliable
        """
    code = self._structarr['patient_orientation'].item()
    if code not in self._patient_orient_codes:
        raise KeyError('Ecat Orientation CODE %d not recognized' % code)
    return self._patient_orient_codes[code]