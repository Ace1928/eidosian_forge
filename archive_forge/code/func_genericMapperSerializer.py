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
def genericMapperSerializer(parent, mapper, mapperId, context, depth):
    dataObject = None
    dataObjectInstance = None
    lookupTableInstance = None
    calls = []
    dependencies = []
    if not hasattr(mapper, 'GetInputDataObject'):
        if context.debugAll:
            print('This mapper does not have GetInputDataObject method')
    else:
        for port in range(mapper.GetNumberOfInputPorts()):
            dataObject = mapper.GetInputDataObject(port, 0)
            if dataObject:
                dataObjectId = '%s-dataset-%d' % (mapperId, port)
                if parent.IsA('vtkActor') and (not mapper.IsA('vtkTexture')):
                    dataObjectInstance = mergeToPolydataSerializer(mapper, dataObject, dataObjectId, context, depth + 1)
                else:
                    dataObjectInstance = serializeInstance(mapper, dataObject, dataObjectId, context, depth + 1)
                if dataObjectInstance:
                    dependencies.append(dataObjectInstance)
                    calls.append(['setInputData', [wrapId(dataObjectId), port]])
    lookupTable = None
    if hasattr(mapper, 'GetLookupTable'):
        lookupTable = mapper.GetLookupTable()
    elif parent.IsA('vtkActor'):
        if context.debugAll:
            print('This mapper actor not have GetLookupTable method')
    if lookupTable:
        lookupTableId = context.getReferenceId(lookupTable)
        lookupTableInstance = serializeInstance(mapper, lookupTable, lookupTableId, context, depth + 1)
        if lookupTableInstance:
            dependencies.append(lookupTableInstance)
            calls.append(['setLookupTable', [wrapId(lookupTableId)]])
    if dataObjectInstance:
        return {'parent': context.getReferenceId(parent), 'id': mapperId, 'properties': {}, 'calls': calls, 'dependencies': dependencies}