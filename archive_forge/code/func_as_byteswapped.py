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
def as_byteswapped(self, endianness=None):
    """Return new object with given ``endianness``

        If big endian, returns a copy of the object. Otherwise raises ValueError.

        Parameters
        ----------
        endianness : None or string, optional
           endian code to which to swap.  None means swap from current
           endianness, and is the default

        Returns
        -------
        wstr : ``MGHHeader``
           ``MGHHeader`` object

        """
    if endianness is None or endian_codes[endianness] != '>':
        raise ValueError('Cannot byteswap MGHHeader - must always be big endian')
    return self.copy()