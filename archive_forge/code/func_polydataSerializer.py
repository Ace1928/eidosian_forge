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
def polydataSerializer(parent, dataset, datasetId, context, depth):
    datasetType = dataset.GetClassName()
    if dataset and dataset.GetPoints():
        properties = {}
        points = getArrayDescription(dataset.GetPoints().GetData(), context)
        points['vtkClass'] = 'vtkPoints'
        properties['points'] = points
        if dataset.GetVerts() and dataset.GetVerts().GetData().GetNumberOfTuples() > 0:
            _verts = getArrayDescription(dataset.GetVerts().GetData(), context)
            properties['verts'] = _verts
            properties['verts']['vtkClass'] = 'vtkCellArray'
        if dataset.GetLines() and dataset.GetLines().GetData().GetNumberOfTuples() > 0:
            _lines = getArrayDescription(dataset.GetLines().GetData(), context)
            properties['lines'] = _lines
            properties['lines']['vtkClass'] = 'vtkCellArray'
        if dataset.GetPolys() and dataset.GetPolys().GetData().GetNumberOfTuples() > 0:
            _polys = getArrayDescription(dataset.GetPolys().GetData(), context)
            properties['polys'] = _polys
            properties['polys']['vtkClass'] = 'vtkCellArray'
        if dataset.GetStrips() and dataset.GetStrips().GetData().GetNumberOfTuples() > 0:
            _strips = getArrayDescription(dataset.GetStrips().GetData(), context)
            properties['strips'] = _strips
            properties['strips']['vtkClass'] = 'vtkCellArray'
        properties['fields'] = []
        extractRequiredFields(properties['fields'], parent, dataset, context)
        return {'parent': context.getReferenceId(parent), 'id': datasetId, 'type': datasetType, 'properties': properties}
    if context.debugAll:
        print('This dataset has no points!')