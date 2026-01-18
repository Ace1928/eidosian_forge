from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def process_edge(edge):
    row = [0 for i in range(self.getNumberOfTriangles())]
    for edgeEmbedding in edge.getEmbeddings():
        tet = edgeEmbedding.getTetrahedron()
        perm = edgeEmbedding.getVertices()
        for face in [2, 3]:
            sign = perm.sign() * (-1) ** face * NTriangulationForPtolemy._sign_of_tetrahedron_face(tet, perm[face])
            index = self._index_of_tetrahedron_face(tet, perm[face])
            row[index] += sign
    return [x / 2 for x in row]