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
def genericProp3DSerializer(parent, prop3D, prop3DId, context, depth):
    instance = genericPropSerializer(parent, prop3D, prop3DId, context, depth)
    if not instance:
        return
    instance['properties'].update({'origin': prop3D.GetOrigin(), 'position': prop3D.GetPosition(), 'scale': prop3D.GetScale(), 'orientation': prop3D.GetOrientation()})
    if prop3D.GetUserMatrix():
        instance['properties'].update({'userMatrix': [prop3D.GetUserMatrix().GetElement(i % 4, i // 4) for i in range(16)]})
    return instance