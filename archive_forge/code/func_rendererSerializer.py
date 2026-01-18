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
def rendererSerializer(parent, instance, objId, context, depth):
    dependencies = []
    viewPropIds = []
    calls = []
    camera = instance.GetActiveCamera()
    cameraId = context.getReferenceId(camera)
    cameraInstance = serializeInstance(instance, camera, cameraId, context, depth + 1)
    if cameraInstance:
        dependencies.append(cameraInstance)
        calls.append(['setActiveCamera', [wrapId(cameraId)]])
    viewPropCollection = instance.GetViewProps()
    for rpIdx in range(viewPropCollection.GetNumberOfItems()):
        viewProp = viewPropCollection.GetItemAsObject(rpIdx)
        viewPropId = context.getReferenceId(viewProp)
        viewPropInstance = serializeInstance(instance, viewProp, viewPropId, context, depth + 1)
        if viewPropInstance:
            dependencies.append(viewPropInstance)
            viewPropIds.append(viewPropId)
    calls += context.buildDependencyCallList('%s-props' % objId, viewPropIds, 'addViewProp', 'removeViewProp')
    return {'parent': context.getReferenceId(parent), 'id': objId, 'type': instance.GetClassName(), 'properties': {'background': instance.GetBackground(), 'background2': instance.GetBackground2(), 'viewport': instance.GetViewport(), 'twoSidedLighting': instance.GetTwoSidedLighting(), 'lightFollowCamera': instance.GetLightFollowCamera(), 'layer': instance.GetLayer(), 'preserveColorBuffer': instance.GetPreserveColorBuffer(), 'preserveDepthBuffer': instance.GetPreserveDepthBuffer(), 'nearClippingPlaneTolerance': instance.GetNearClippingPlaneTolerance(), 'clippingRangeExpansion': instance.GetClippingRangeExpansion(), 'useShadows': instance.GetUseShadows(), 'useDepthPeeling': instance.GetUseDepthPeeling(), 'occlusionRatio': instance.GetOcclusionRatio(), 'maximumNumberOfPeels': instance.GetMaximumNumberOfPeels(), 'interactive': instance.GetInteractive()}, 'dependencies': dependencies, 'calls': calls}