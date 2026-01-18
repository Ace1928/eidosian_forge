from __future__ import annotations
import numpy as np
from .arrayproxy import ArrayProxy
from .arraywriters import ArrayWriter, WriterError, get_slope_inter, make_array_writer
from .batteryrunners import Report
from .fileholders import copy_file_map
from .spatialimages import HeaderDataError, HeaderTypeError, SpatialHeader, SpatialImage
from .volumeutils import (
from .wrapstruct import LabeledWrapStruct
def set_data_offset(self, offset):
    """Set offset into data file to read data"""
    self._structarr['vox_offset'] = offset