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
@classmethod
def diagnose_binaryblock(klass, binaryblock, endianness=None):
    if endianness is not None and endian_codes[endianness] != '>':
        raise ValueError('MGHHeader must always be big endian')
    wstr = klass(binaryblock, check=False)
    battrun = BatteryRunner(klass._get_checks())
    reports = battrun.check_only(wstr)
    return '\n'.join([report.message for report in reports if report.message])