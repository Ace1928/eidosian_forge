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
def renderWindowSerializer(parent, instance, objId, context, depth):
    dependencies = []
    rendererIds = []
    rendererCollection = instance.GetRenderers()
    for rIdx in range(rendererCollection.GetNumberOfItems()):
        renderer = rendererCollection.GetItemAsObject(rIdx)
        rendererId = context.getReferenceId(renderer)
        rendererInstance = serializeInstance(instance, renderer, rendererId, context, depth + 1)
        if rendererInstance:
            dependencies.append(rendererInstance)
            rendererIds.append(rendererId)
    calls = context.buildDependencyCallList(objId, rendererIds, 'addRenderer', 'removeRenderer')
    return {'parent': context.getReferenceId(parent), 'id': objId, 'type': instance.GetClassName(), 'properties': {'numberOfLayers': instance.GetNumberOfLayers()}, 'dependencies': dependencies, 'calls': calls, 'mtime': instance.GetMTime()}