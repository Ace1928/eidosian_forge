import base64
import hashlib
import io
import struct
import time
import zipfile
from vtk.vtkCommonCore import vtkTypeInt32Array, vtkTypeUInt32Array
from vtk.vtkCommonDataModel import vtkDataObject
from vtk.vtkFiltersGeometry import (
from vtk.vtkRenderingCore import vtkColorTransferFunction
from .enums import TextPosition
def getRangeInfo(array, component):
    r = array.GetRange(component)
    compRange = {}
    compRange['min'] = r[0]
    compRange['max'] = r[1]
    compRange['component'] = array.GetComponentName(component)
    return compRange