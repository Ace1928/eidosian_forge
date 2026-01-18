from collections import OrderedDict
from ... import sage_helper
def build_vertices(self):
    """
        Build the 0 skeleton.
        """
    vertices = []
    corners = OrderedDict([[(T, v), Corner(T, v)] for T in self.triangles for v in range(3)])
    while corners:
        C0 = next(iter(corners.values()))
        vertex = [C0]
        C = C0.next_corner()
        while C != C0:
            vertex.append(C)
            C = C.next_corner()
        V = Vertex(vertex)
        for C in vertex:
            corners.pop((C.triangle, C.vertex))
            C.triangle.vertices[C.vertex] = V
        vertices.append(V)
    self.vertices = vertices
    self._vertex_containing_corner = dict([((C.triangle, C.vertex), V) for V in vertices for C in V.corners])