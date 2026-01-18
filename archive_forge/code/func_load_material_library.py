import os
import pyglet
from pyglet.gl import GL_TRIANGLES
from pyglet.util import asstr
from .. import Model, Material, MaterialGroup, TexturedMaterialGroup
from . import ModelDecodeException, ModelDecoder
def load_material_library(filename):
    file = open(filename, 'r')
    name = None
    diffuse = [1.0, 1.0, 1.0]
    ambient = [1.0, 1.0, 1.0]
    specular = [1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0]
    shininess = 100.0
    opacity = 1.0
    texture_name = None
    matlib = {}
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'newmtl':
            if name is not None:
                for item in (diffuse, ambient, specular, emission):
                    item.append(opacity)
                matlib[name] = Material(name, diffuse, ambient, specular, emission, shininess, texture_name)
            name = values[1]
        elif name is None:
            raise ModelDecodeException(f'Expected "newmtl" in {filename}')
        try:
            if values[0] == 'Kd':
                diffuse = list(map(float, values[1:]))
            elif values[0] == 'Ka':
                ambient = list(map(float, values[1:]))
            elif values[0] == 'Ks':
                specular = list(map(float, values[1:]))
            elif values[0] == 'Ke':
                emission = list(map(float, values[1:]))
            elif values[0] == 'Ns':
                shininess = float(values[1])
                shininess = shininess * 128 / 1000
            elif values[0] == 'd':
                opacity = float(values[1])
            elif values[0] == 'map_Kd':
                texture_name = values[1]
        except BaseException as ex:
            raise ModelDecodeException('Parsing error in {0}.'.format((filename, ex)))
    file.close()
    for item in (diffuse, ambient, specular, emission):
        item.append(opacity)
    matlib[name] = Material(name, diffuse, ambient, specular, emission, shininess, texture_name)
    return matlib