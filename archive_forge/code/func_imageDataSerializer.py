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
def imageDataSerializer(parent, dataset, datasetId, context, depth):
    datasetType = dataset.GetClassName()
    if hasattr(dataset, 'GetDirectionMatrix'):
        direction = [dataset.GetDirectionMatrix().GetElement(0, i) for i in range(9)]
    else:
        direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    arrays = []
    extractRequiredFields(arrays, parent, dataset, context)
    return {'parent': context.getReferenceId(parent), 'id': datasetId, 'type': datasetType, 'properties': {'spacing': dataset.GetSpacing(), 'origin': dataset.GetOrigin(), 'dimensions': dataset.GetDimensions(), 'direction': direction}, 'arrays': arrays}