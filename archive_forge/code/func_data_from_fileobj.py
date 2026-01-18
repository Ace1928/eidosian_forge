from os.path import splitext
import numpy as np
from ..affines import from_matvec, voxel_sizes
from ..arrayproxy import ArrayProxy, reshape_dataobj
from ..batteryrunners import BatteryRunner, Report
from ..filebasedimages import SerializableImage
from ..fileholders import FileHolder
from ..filename_parser import _stringify_path
from ..openers import ImageOpener
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..volumeutils import Recoder, array_from_file, array_to_file, endian_codes
from ..wrapstruct import LabeledWrapStruct
def data_from_fileobj(self, fileobj):
    """Read data array from `fileobj`

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           data array
        """
    dtype = self.get_data_dtype()
    shape = self.get_data_shape()
    offset = self.get_data_offset()
    return array_from_file(shape, dtype, fileobj, offset)