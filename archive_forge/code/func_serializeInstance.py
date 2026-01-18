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
def serializeInstance(parent, instance, instanceId, context, depth):
    instanceType = instance.GetClassName()
    serializer = SERIALIZERS[instanceType] if instanceType in SERIALIZERS else None
    if serializer:
        return serializer(parent, instance, instanceId, context, depth)
    if context.debugSerializers:
        print('%s!!!No serializer for %s with id %s' % (pad(depth), instanceType, instanceId))