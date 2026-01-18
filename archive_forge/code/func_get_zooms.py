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
def get_zooms(self):
    """Get zooms from header

        Returns the spacing of voxels in the x, y, and z dimensions.
        For four-dimensional files, a fourth zoom is included, equal to the
        repetition time (TR) in ms (see `The MGH/MGZ Volume Format
        <mghformat>`_).

        To access only the spatial zooms, use `hdr['delta']`.

        Returns
        -------
        z : tuple
           tuple of header zoom values

        .. _mghformat: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat#line-82
        """
    tzoom = (self['tr'],) if self._ndims() > 3 else ()
    return tuple(self._structarr['delta']) + tzoom