from collections import OrderedDict
from ... import sage_helper
def glue_triangles(self, T0, e0, T1, e1):
    """
        T0, T1 are triangles, and e0, e1 specify a gluing in one of two
        ways: as a list of vertices of the triangle, or as the vertex
        opposite the edge being glued.  In the latter case, the assumption
        is that the gluing preserves orientation.

        Returns the newly created edge.
        """
    if not hasattr(e0, '__iter__'):
        e0 = oriented_edges_of_triangle[e0]
        a, b = oriented_edges_of_triangle[e1]
        e1 = (b, a)
        S0, S1 = (Side(T0, e0), Side(T1, e1))
        E = Edge(sides=(S0, S1))
        T0.edges[opposite_vertex_from_edge_dict[e0]] = E
        T1.edges[opposite_vertex_from_edge_dict[e1]] = E
        self.edges.append(E)
        return E