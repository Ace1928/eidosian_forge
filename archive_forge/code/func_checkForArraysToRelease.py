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
def checkForArraysToRelease(self, timeWindow=20):
    cutOffTime = time.time() - timeWindow
    shasToDelete = []
    for sha in self.dataArrayCache:
        record = self.dataArrayCache[sha]
        array = record['array']
        count = array.GetReferenceCount()
        if count == 1 and record['ts'] < cutOffTime:
            shasToDelete.append(sha)
    for sha in shasToDelete:
        del self.dataArrayCache[sha]