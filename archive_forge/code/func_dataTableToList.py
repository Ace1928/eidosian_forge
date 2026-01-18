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
def dataTableToList(dataTable):
    dataType = arrayTypesMapping[dataTable.GetDataType()]
    elementSize = struct.calcsize(dataType)
    nbValues = dataTable.GetNumberOfValues()
    nbComponents = dataTable.GetNumberOfComponents()
    nbytes = elementSize * nbValues
    if dataType != ' ':
        with io.BytesIO(buffer(dataTable)) as stream:
            data = list(struct.unpack(dataType * nbValues, stream.read(nbytes)))
        return [data[idx * nbComponents:(idx + 1) * nbComponents] for idx in range(nbValues // nbComponents)]