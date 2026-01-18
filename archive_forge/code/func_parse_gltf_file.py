import json
import struct
import pyglet
from pyglet.gl import GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_FLOAT
from pyglet.gl import GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER, GL_ARRAY_BUFFER, GL_TRIANGLES
from .. import Model, Material, MaterialGroup
from . import ModelDecodeException, ModelDecoder
def parse_gltf_file(file, filename, batch):
    if file is None:
        file = pyglet.resource.file(filename, 'r')
    elif file.mode != 'r':
        file.close()
        file = pyglet.resource.file(filename, 'r')
    try:
        data = json.load(file)
    except json.JSONDecodeError:
        raise ModelDecodeException('Json error. Does not appear to be a valid glTF file.')
    finally:
        file.close()
    if 'asset' not in data:
        raise ModelDecodeException('Not a valid glTF file. Asset property not found.')
    elif float(data['asset']['version']) < 2.0:
        raise ModelDecodeException('Only glTF 2.0+ models are supported')
    buffers = dict()
    buffer_views = dict()
    accessors = dict()
    materials = dict()
    for i, item in enumerate(data.get('buffers', [])):
        buffers[i] = Buffer(item['byteLength'], item['uri'])
    for i, item in enumerate(data.get('bufferViews', [])):
        buffer_index = item['buffer']
        buffer = buffers[buffer_index]
        offset = item.get('byteOffset', 0)
        length = item.get('byteLength')
        target = item.get('target')
        stride = item.get('byteStride', 1)
        buffer_views[i] = BufferView(buffer, offset, length, target, stride)
    for i, item in enumerate(data.get('accessors', [])):
        buf_view_index = item.get('bufferView')
        buf_view = buffer_views[buf_view_index]
        offset = item.get('byteOffset', 0)
        comp_type = item.get('componentType')
        count = item.get('count')
        maxi = item.get('max')
        mini = item.get('min')
        acc_type = item.get('type')
        sparse = item.get('sparse', None)
        accessors[i] = Accessor(buf_view, offset, comp_type, count, maxi, mini, acc_type, sparse)
    vertex_lists = []
    for mesh_data in data.get('meshes'):
        for primitive in mesh_data.get('primitives', []):
            indices = None
            attribute_list = []
            count = 0
            for attribute_type, i in primitive['attributes'].items():
                accessor = accessors[i]
                attrib = _attributes[attribute_type]
                if not attrib:
                    continue
                attrib_size = _accessor_type_sizes[accessor.type]
                pyglet_type = _pyglet_types[accessor.component_type]
                pyglet_fmt = '{0}{1}{2}'.format(attrib, attrib_size, pyglet_type)
                count = accessor.count
                struct_fmt = str(count * attrib_size) + _struct_types[accessor.component_type]
                array = struct.unpack('<' + struct_fmt, accessor.read())
                attribute_list.append((pyglet_fmt, array))
            if 'indices' in primitive:
                indices_index = primitive.get('indices')
                accessor = accessors[indices_index]
                attrib_size = _accessor_type_sizes[accessor.type]
                fmt = str(accessor.count * attrib_size) + _struct_types[accessor.component_type]
                indices = struct.unpack('<' + fmt, accessor.read())
            diffuse = [1.0, 1.0, 1.0]
            ambient = [1.0, 1.0, 1.0]
            specular = [1.0, 1.0, 1.0]
            emission = [0.0, 0.0, 0.0]
            shininess = 100.0
            opacity = 1.0
            material = Material('Default', diffuse, ambient, specular, emission, shininess, opacity)
            group = MaterialGroup(material=material)
            if indices:
                vlist = batch.add_indexed(count, GL_TRIANGLES, group, indices, *attribute_list)
            else:
                vlist = batch.add(count, GL_TRIANGLES, group, *attribute_list)
            vertex_lists.append(vlist)
    return vertex_lists