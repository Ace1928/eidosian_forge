import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_mobius_strip(num_segments=100, radius=1.0, width=0.1):
    """
    Construct a Möbius strip using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a Möbius strip with detailed vertex and color definitions, using structured arrays for optimal data management.

    Parameters:
    - num_segments (int): The number of segments used to approximate the Möbius strip.
    - radius (float): The central radius of the Möbius strip.
    - width (float): The width of the strip.

    Returns:
    - GeomNode: A geometry node containing the constructed Möbius strip geometry.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('mobius_strip_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    vertices = np.zeros((num_segments * 2, 3), dtype=np.float32)
    colors = np.zeros((num_segments * 2, 4), dtype=np.float32)
    for i in range(num_segments):
        t = 2 * np.pi * i / num_segments
        for j in [-1, 1]:
            index = i * 2 + (j + 1) // 2
            x = (radius + j * width * np.cos(t / 2)) * np.cos(t)
            y = (radius + j * width * np.cos(t / 2)) * np.sin(t)
            z = j * width * np.sin(t / 2)
            vertices[index] = [x, y, z]
            colors[index] = [np.sin(t), np.cos(t), np.abs(np.sin(t / 2)), 1.0]
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = np.array([(i, (i + 1) % (num_segments * 2), (i + 2) % (num_segments * 2)) if i % 2 == 0 else ((i + 1) % (num_segments * 2), i, (i + 2) % (num_segments * 2)) for i in range(num_segments * 2 - 2)], dtype=np.int32)
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('mobius_strip_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node