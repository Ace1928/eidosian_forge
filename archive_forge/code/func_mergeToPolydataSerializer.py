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
def mergeToPolydataSerializer(parent, dataObject, dataObjectId, context, depth):
    dataset = None
    if dataObject.IsA('vtkCompositeDataSet'):
        gf = vtkCompositeDataGeometryFilter()
        gf.SetInputData(dataObject)
        gf.Update()
        dataset = gf.GetOutput()
    elif not dataObject.IsA('vtkPolyData'):
        gf = vtkGeometryFilter()
        gf.SetInputData(dataObject)
        gf.Update()
        dataset = gf.GetOutput()
    else:
        dataset = dataObject
    return polydataSerializer(parent, dataset, dataObjectId, context, depth)