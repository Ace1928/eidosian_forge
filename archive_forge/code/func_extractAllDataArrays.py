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
def extractAllDataArrays(extractedFields, dataset, context):
    pointData = dataset.GetPointData()
    for id_arr in range(pointData.GetNumberOfArrays()):
        arrayMeta = getArrayDescription(pointData.GetArray(id_arr), context)
        if arrayMeta:
            arrayMeta['location'] = 'pointData'
            extractedFields.append(arrayMeta)
    cellData = dataset.GetCellData()
    for id_arr in range(cellData.GetNumberOfArrays()):
        arrayMeta = getArrayDescription(cellData.GetArray(id_arr), context)
        if arrayMeta:
            arrayMeta['location'] = 'cellData'
            extractedFields.append(arrayMeta)
    fieldData = dataset.GetCellData()
    for id_arr in range(fieldData.GetNumberOfArrays()):
        arrayMeta = getArrayDescription(fieldData.GetArray(id_arr), context)
        if arrayMeta:
            arrayMeta['location'] = 'fieldData'
            extractedFields.append(arrayMeta)