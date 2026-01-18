import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_circle_sheet_with_vertex_data():
    """
    Construct a circle sheet using vertex data and geom nodes, employing structured arrays for vertex and color data.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('circle_sheet_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    num_segments = 32
    radius = 1.0
    angle_increment = 2 * np.pi / num_segments
    vertices = np.array([[np.cos(i * angle_increment) * radius, np.sin(i * angle_increment) * radius, 0] for i in range(num_segments)], dtype=np.float32)
    colors = np.array([[np.cos(i * angle_increment), np.sin(i * angle_increment), 0.5, 1] for i in range(num_segments)], dtype=np.float32)
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = np.array([[i, (i + 1) % num_segments, num_segments] for i in range(num_segments)], dtype=np.int32)
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('circle_sheet_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node