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
def get_vox2ras_tkr(self):
    """Get the vox2ras-tkr transform. See "Torig" here:
        https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        """
    ds = self._structarr['delta']
    ns = self._structarr['dims'][:3] * ds / 2.0
    v2rtkr = np.array([[-ds[0], 0, 0, ns[0]], [0, 0, ds[2], -ns[2]], [0, -ds[1], 0, ns[1]], [0, 0, 0, 1]], dtype=np.float32)
    return v2rtkr