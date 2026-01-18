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
def genericVolumeMapperSerializer(parent, mapper, mapperId, context, depth):
    instance = genericMapperSerializer(parent, mapper, mapperId, context, depth)
    if not instance:
        return
    imageSampleDistance = mapper.GetImageSampleDistance() if hasattr(mapper, 'GetImageSampleDistance') else 1
    instance['type'] = mapper.GetClassName()
    instance['properties'].update({'sampleDistance': mapper.GetSampleDistance(), 'imageSampleDistance': imageSampleDistance, 'autoAdjustSampleDistances': mapper.GetAutoAdjustSampleDistances(), 'blendMode': mapper.GetBlendMode()})
    return instance