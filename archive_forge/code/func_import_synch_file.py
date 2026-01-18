import json
import re
import struct
import zipfile
import vtk
from .synchronizable_serializer import arrayTypesMapping
def import_synch_file(filename):
    with zipfile.ZipFile(filename, 'r') as zf:
        scene = json.loads(zf.read('index.json').decode())
        scene['properties']['numberOfLayers'] = 1
        renwin = generic_builder(scene, zf)
    return renwin