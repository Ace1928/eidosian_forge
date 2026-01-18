from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def get_generators(tet):
    """
            Given a tetrahedron, return for each face which inbound
            or outbound generator it belongs to.
            """
    return [get_generator(tet, face) for face in range(4)]