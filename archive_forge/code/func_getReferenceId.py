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
def getReferenceId(self, instance):
    if not self.idRoot or (hasattr(instance, 'IsA') and instance.IsA('vtkCamera')):
        return getReferenceId(instance)
    else:
        return self.idRoot + getReferenceId(instance)