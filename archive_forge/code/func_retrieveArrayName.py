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
def retrieveArrayName(mapper_instance, scalar_mode):
    colorArrayName = None
    try:
        ds = [deps for deps in mapper_instance['dependencies'] if deps['id'].endswith('dataset')][0]
        location = 'pointData' if scalar_mode in (1, 3) else 'cellData'
        for arrayMeta in ds['properties']['fields']:
            if arrayMeta['location'] == location and arrayMeta.get('registration', None) == 'setScalars':
                colorArrayName = arrayMeta['name']
    except Exception:
        pass
    return colorArrayName