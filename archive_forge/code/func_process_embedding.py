from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def process_embedding(embedding):
    face = embedding.getFace()
    tet = self.tetrahedronIndex(embedding.getTetrahedron())
    return 's_%d_%d' % (face, tet)