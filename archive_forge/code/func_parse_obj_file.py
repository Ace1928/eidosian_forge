import os
import pyglet
from pyglet.gl import GL_TRIANGLES
from pyglet.util import asstr
from .. import Model, Material, MaterialGroup, TexturedMaterialGroup
from . import ModelDecodeException, ModelDecoder
def parse_obj_file(filename, file=None):
    materials = {}
    mesh_list = []
    location = os.path.dirname(filename)
    try:
        if file is None:
            with open(filename, 'r') as f:
                file_contents = f.read()
        else:
            file_contents = asstr(file.read())
    except (UnicodeDecodeError, OSError):
        raise ModelDecodeException
    material = None
    mesh = None
    vertices = [[0.0, 0.0, 0.0]]
    normals = [[0.0, 0.0, 0.0]]
    tex_coords = [[0.0, 0.0]]
    diffuse = [1.0, 1.0, 1.0, 1.0]
    ambient = [1.0, 1.0, 1.0, 1.0]
    specular = [1.0, 1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 100.0
    default_material = Material('Default', diffuse, ambient, specular, emission, shininess)
    for line in file_contents.splitlines():
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            vertices.append(list(map(float, values[1:4])))
        elif values[0] == 'vn':
            normals.append(list(map(float, values[1:4])))
        elif values[0] == 'vt':
            tex_coords.append(list(map(float, values[1:3])))
        elif values[0] == 'mtllib':
            material_abspath = os.path.join(location, values[1])
            materials = load_material_library(filename=material_abspath)
        elif values[0] in ('usemtl', 'usemat'):
            material = materials.get(values[1])
            if mesh is not None:
                mesh.material = material
        elif values[0] == 'o':
            mesh = Mesh(name=values[1])
            mesh.material = default_material
            mesh_list.append(mesh)
        elif values[0] == 'f':
            if mesh is None:
                mesh = Mesh(name='')
                mesh_list.append(mesh)
            if material is None:
                material = default_material
            if mesh.material is None:
                mesh.material = material
            n1 = None
            nlast = None
            t1 = None
            tlast = None
            v1 = None
            vlast = None
            for i, v in enumerate(values[1:]):
                v_i, t_i, n_i = (list(map(int, [j or 0 for j in v.split('/')])) + [0, 0])[:3]
                if v_i < 0:
                    v_i += len(vertices) - 1
                if t_i < 0:
                    t_i += len(tex_coords) - 1
                if n_i < 0:
                    n_i += len(normals) - 1
                mesh.normals += normals[n_i]
                mesh.tex_coords += tex_coords[t_i]
                mesh.vertices += vertices[v_i]
                if i >= 3:
                    mesh.normals += n1 + nlast
                    mesh.tex_coords += t1 + tlast
                    mesh.vertices += v1 + vlast
                if i == 0:
                    n1 = normals[n_i]
                    t1 = tex_coords[t_i]
                    v1 = vertices[v_i]
                nlast = normals[n_i]
                tlast = tex_coords[t_i]
                vlast = vertices[v_i]
    return mesh_list