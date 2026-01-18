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
def getCachedDataArray(self, pMd5, binary=False, compression=False):
    cacheObj = self.dataArrayCache[pMd5]
    array = cacheObj['array']
    cacheTime = cacheObj['mTime']
    if cacheTime != array.GetMTime():
        if context.debugAll:
            print(' ***** ERROR: you asked for an old cache key! ***** ')
    if array.GetDataType() in (12, 16, 17):
        arraySize = array.GetNumberOfTuples() * array.GetNumberOfComponents()
        if array.GetDataType() in (12, 17):
            newArray = vtkTypeUInt32Array()
        else:
            newArray = vtkTypeInt32Array()
        newArray.SetNumberOfTuples(arraySize)
        for i in range(arraySize):
            newArray.SetValue(i, -1 if array.GetValue(i) < 0 else array.GetValue(i))
        pBuffer = buffer(newArray)
    else:
        pBuffer = buffer(array)
    if binary:
        return pBuffer.tobytes() if not compression else zipCompression(pMd5, pBuffer.tobytes())
    return base64Encode(pBuffer if not compression else zipCompression(pMd5, pBuffer.tobytes()))