import json
import re
import struct
import zipfile
import vtk
from .synchronizable_serializer import arrayTypesMapping
def poly_data_builder(state, zf, register):
    instance = vtk.vtkPolyData()
    register.update({state['id']: instance})
    if 'points' in state['properties']:
        points = state['properties']['points']
        vtkpoints = vtk.vtkPoints()
        points_data_arr = ARRAY_TYPES[points['dataType']]()
        fill_array(points_data_arr, points, zf)
        vtkpoints.SetData(points_data_arr)
        instance.SetPoints(vtkpoints)
    for cell_type in ['verts', 'lines', 'polys', 'strips']:
        if cell_type in state['properties']:
            cell_arr = vtk.vtkCellArray()
            cell_data_arr = vtk.vtkIdTypeArray()
            fill_array(cell_data_arr, state['properties'][cell_type], zf)
            cell_arr.ImportLegacyFormat(cell_data_arr)
            getattr(instance, 'Set' + capitalize(cell_type))(cell_arr)
    fields = state['properties']['fields']
    for dataset in fields:
        data_arr = ARRAY_TYPES[dataset['dataType']]()
        fill_array(data_arr, dataset, zf)
        location = getattr(instance, 'Get' + capitalize(dataset['location']))()
        getattr(location, capitalize(dataset.get('registration', 'addArray')))(data_arr)