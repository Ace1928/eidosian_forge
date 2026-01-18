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
def genericActorSerializer(parent, actor, actorId, context, depth):
    instance = genericProp3DSerializer(parent, actor, actorId, context, depth)
    if not instance:
        return
    instance['properties'].update({'forceOpaque': actor.GetForceOpaque(), 'forceTranslucent': actor.GetForceTranslucent()})
    if actor.IsA('vtkFollower'):
        camera = actor.GetCamera()
        cameraId = context.getReferenceId(camera)
        cameraInstance = serializeInstance(actor, camera, cameraId, context, depth + 1)
        if cameraInstance:
            instance['dependencies'].append(cameraInstance)
            instance['calls'].append(['setCamera', [wrapId(cameraId)]])
    return instance