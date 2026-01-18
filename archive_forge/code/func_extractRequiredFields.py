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
def extractRequiredFields(extractedFields, parent, dataset, context, requestedFields=['Normals', 'TCoords']):
    if any((parent.IsA(cls) for cls in ['vtkMapper', 'vtkVolumeMapper', 'vtkImageSliceMapper', 'vtkTexture'])):
        if parent.IsA('vtkAbstractMapper'):
            scalarVisibility = 1 if not hasattr(parent, 'GetScalarVisibility') else parent.GetScalarVisibility()
            scalars, cell_flag = getScalars(parent, dataset)
            if context.serializeAllDataArrays:
                extractAllDataArrays(extractedFields, dataset, context)
                if scalars:
                    for arrayMeta in extractedFields:
                        if arrayMeta['name'] == scalars.GetName():
                            arrayMeta['registration'] = 'setScalars'
            elif scalars and scalarVisibility and (not context.serializeAllDataArrays):
                arrayMeta = getArrayDescription(scalars, context)
                if cell_flag == 0:
                    arrayMeta['location'] = 'pointData'
                elif cell_flag == 1:
                    arrayMeta['location'] = 'cellData'
                else:
                    raise NotImplementedError('Scalars on field data not handled')
                arrayMeta['registration'] = 'setScalars'
                extractedFields.append(arrayMeta)
        elif dataset.GetPointData().GetScalars():
            arrayMeta = getArrayDescription(dataset.GetPointData().GetScalars(), context)
            arrayMeta['location'] = 'pointData'
            arrayMeta['registration'] = 'setScalars'
            extractedFields.append(arrayMeta)
    if parent.IsA('vtkGlyph3DMapper') and (not context.serializeAllDataArrays):
        scaleArrayName = parent.GetInputArrayInformation(parent.SCALE).Get(vtkDataObject.FIELD_NAME())
        if scaleArrayName is not None and scaleArrayName not in [field['name'] for field in extractedFields]:
            arrayMeta = getArrayDescription(dataset.GetPointData().GetAbstractArray(scaleArrayName), context)
            if arrayMeta is not None:
                arrayMeta['location'] = 'pointData'
                arrayMeta['registration'] = 'addArray'
                extractedFields.append(arrayMeta)
        scaleOrientationArrayName = parent.GetInputArrayInformation(parent.ORIENTATION).Get(vtkDataObject.FIELD_NAME())
        if scaleOrientationArrayName is not None and scaleOrientationArrayName not in [field['name'] for field in extractedFields]:
            arrayMeta = getArrayDescription(dataset.GetPointData().GetAbstractArray(scaleOrientationArrayName), context)
            if arrayMeta is not None:
                arrayMeta['location'] = 'pointData'
                arrayMeta['registration'] = 'addArray'
                extractedFields.append(arrayMeta)
    if 'Normals' in requestedFields:
        normals = dataset.GetPointData().GetNormals()
        if normals:
            arrayMeta = getArrayDescription(normals, context)
            if arrayMeta:
                arrayMeta['location'] = 'pointData'
                arrayMeta['registration'] = 'setNormals'
                extractedFields.append(arrayMeta)
    if 'TCoords' in requestedFields:
        tcoords = dataset.GetPointData().GetTCoords()
        if tcoords:
            arrayMeta = getArrayDescription(tcoords, context)
            if arrayMeta:
                arrayMeta['location'] = 'pointData'
                arrayMeta['registration'] = 'setTCoords'
                extractedFields.append(arrayMeta)