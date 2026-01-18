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
def getScalars(mapper, dataset):
    scalars = None
    cell_flag = 0
    scalar_mode = mapper.GetScalarMode()
    array_access_mode = mapper.GetArrayAccessMode()
    array_id = mapper.GetArrayId()
    array_name = mapper.GetArrayName()
    pd = dataset.GetPointData()
    cd = dataset.GetCellData()
    fd = dataset.GetFieldData()
    if scalar_mode == 0:
        scalars = pd.GetScalars()
        cell_flag = 0
        if scalars is None:
            scalars = cd.GetScalars()
            cell_flag = 1
    elif scalar_mode == 1:
        scalars = pd.GetScalars()
        cell_flag = 0
    elif scalar_mode == 2:
        scalars = cd.GetScalars()
        cell_flag = 1
    elif scalar_mode == 3:
        if array_access_mode == 0:
            scalars = pd.GetAbstractArray(array_id)
        else:
            scalars = pd.GetAbstractArray(array_name)
        cell_flag = 0
    elif scalar_mode == 4:
        if array_access_mode == 0:
            scalars = cd.GetAbstractArray(array_id)
        else:
            scalars = cd.GetAbstractArray(array_name)
        cell_flag = 1
    else:
        if array_access_mode == 0:
            scalars = fd.GetAbstractArray(array_id)
        else:
            scalars = fd.GetAbstractArray(array_name)
        cell_flag = 2
    return (scalars, cell_flag)