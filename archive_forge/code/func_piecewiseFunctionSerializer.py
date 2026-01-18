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
def piecewiseFunctionSerializer(parent, instance, objId, context, depth):
    nodes = []
    for i in range(instance.GetSize()):
        node = [0, 0, 0, 0]
        instance.GetNodeValue(i, node)
        nodes.append(node)
    return {'parent': context.getReferenceId(parent), 'id': objId, 'type': instance.GetClassName(), 'properties': {'clamping': instance.GetClamping(), 'allowDuplicateScalars': instance.GetAllowDuplicateScalars(), 'nodes': nodes}}