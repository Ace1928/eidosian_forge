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
def genericPropSerializer(parent, prop, popId, context, depth):
    mapperInstance = None
    propertyInstance = None
    calls = []
    dependencies = []
    mapper = None
    if not hasattr(prop, 'GetMapper'):
        if context.debugAll:
            print('This volume does not have a GetMapper method')
    else:
        mapper = prop.GetMapper()
    if mapper:
        mapperId = context.getReferenceId(mapper)
        mapperInstance = serializeInstance(prop, mapper, mapperId, context, depth + 1)
        if mapperInstance:
            dependencies.append(mapperInstance)
            calls.append(['setMapper', [wrapId(mapperId)]])
    properties = None
    if hasattr(prop, 'GetProperty'):
        properties = prop.GetProperty()
    elif context.debugAll:
        print('This image does not have a GetProperty method')
    if properties:
        propId = context.getReferenceId(properties)
        propertyInstance = serializeInstance(prop, properties, propId, context, depth + 1)
        if propertyInstance:
            dependencies.append(propertyInstance)
            calls.append(['setProperty', [wrapId(propId)]])
    texture = None
    if hasattr(prop, 'GetTexture'):
        texture = prop.GetTexture()
    if texture:
        textureId = context.getReferenceId(texture)
        textureInstance = serializeInstance(prop, texture, textureId, context, depth + 1)
        if textureInstance:
            dependencies.append(textureInstance)
            calls.append(['addTexture', [wrapId(textureId)]])
    return {'parent': context.getReferenceId(parent), 'id': popId, 'type': prop.GetClassName(), 'properties': {'visibility': prop.GetVisibility(), 'pickable': prop.GetPickable(), 'dragable': prop.GetDragable(), 'useBounds': prop.GetUseBounds()}, 'calls': calls, 'dependencies': dependencies}