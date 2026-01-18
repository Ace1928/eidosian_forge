from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def build_weights(self, manifold):
    """
        Use self.QuadWeights self.QuadTypes vector to construct
        self.Weights and self.EdgeWeights.  The vector self.Weights has size
        7T and gives the weights of triangles and quads in each 3-simplex.
        In each bank of 7 weights, the first 4 are triangle weights and the
        last 3 are quad weights.
        """
    self.Weights = Vector(7 * self.Size)
    eqns = []
    constants = []
    edge_matrix = []
    for edge in manifold.Edges:
        edge_row = Vector(7 * len(manifold))
        corner = edge.Corners[0]
        j = corner.Tetrahedron.Index
        edge_row[7 * j:7 * j + 4] = MeetsTri[corner.Subsimplex]
        if not self.Coefficients[j] == -1:
            edge_row[7 * j + 4:7 * j + 7] = MeetsQuad[corner.Subsimplex]
        else:
            edge_row[7 * j + 4:7 * j + 7] = MeetsOct[corner.Subsimplex]
        edge_matrix.append(edge_row)
        for i in range(len(edge.Corners) - 1):
            j = edge.Corners[i].Tetrahedron.Index
            k = edge.Corners[i + 1].Tetrahedron.Index
            row = Vector(4 * len(manifold))
            row[4 * j:4 * j + 4] = MeetsTri[edge.Corners[i].Subsimplex]
            row[4 * k:4 * k + 4] -= MeetsTri[edge.Corners[i + 1].Subsimplex]
            eqns.append(row)
            c = 0
            if self.Coefficients[k] == -1:
                c = MeetsOct[edge.Corners[i + 1].Subsimplex][self.Quadtypes[k]]
            elif MeetsQuad[edge.Corners[i + 1].Subsimplex][self.Quadtypes[k]]:
                c = self.Coefficients[k]
            if self.Coefficients[j] == -1:
                c -= MeetsOct[edge.Corners[i].Subsimplex][self.Quadtypes[j]]
            elif MeetsQuad[edge.Corners[i].Subsimplex][self.Quadtypes[j]]:
                c -= self.Coefficients[j]
            constants.append(c)
        for vertex in manifold.Vertices:
            eqns.append(vertex.IncidenceVector)
            constants.append(0)
    A = Matrix(eqns)
    b = Vector(constants)
    x = A.solve(b)
    for vertex in manifold.Vertices:
        vert_vec = vertex.IncidenceVector
        m = min([x[i] for i, w in enumerate(vert_vec) if w])
        x -= Vector(m * vert_vec)
    for i in range(len(manifold)):
        for j in range(4):
            v = x[4 * i + j]
            assert int(v) == v
            self.Weights[7 * i + j] = int(v)
        if not self.Coefficients[i] == -1:
            self.Weights[7 * i + 4:7 * i + 7] = self.Coefficients[i] * QuadWeights[int(self.Quadtypes[i])]
        else:
            self.Weights[7 * i + 4:7 * i + 7] = QuadWeights[int(self.Quadtypes[i])]
    self.EdgeWeights = Matrix(edge_matrix).dot(self.Weights)