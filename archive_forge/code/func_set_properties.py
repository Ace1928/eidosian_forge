import json
import re
import struct
import zipfile
import vtk
from .synchronizable_serializer import arrayTypesMapping
def set_properties(instance, properties):
    for k, v in properties.items():
        fn = getattr(instance, 'Set' + capitalize(k), None)
        if fn:
            fn(v)