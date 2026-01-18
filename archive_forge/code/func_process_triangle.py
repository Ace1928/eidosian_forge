from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def process_triangle(triangle):
    row = [0 for i in range(self.getNumberOfTetrahedra())]
    for i in range(2):
        index = self.tetrahedronIndex(triangle.getEmbedding(i).getTetrahedron())
        row[index] += (-1) ** i
    return row