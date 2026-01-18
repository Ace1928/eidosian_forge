import numpy as np
from panda3d.core import (
from panda3d.core import RenderModeAttrib
def construct_rectangular_prism():
    """
    Construct a rectangular prism using vertex data and geom nodes, with detailed vertex and color definitions.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData('rectangular_prism_vertices_and_colors', vertex_format, Geom.UHStatic)
    vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
    color_writer = GeomVertexWriter(vertex_data, 'color')
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float32)
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]], dtype=np.float32)
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)
    triangle_indices = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3], [0, 4, 7], [0, 7, 3], [1, 5, 6], [1, 6, 2]], dtype=np.int32)
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode('rectangular_prism_geom_node')
    geometry_node.addGeom(geometry)
    return geometry_node