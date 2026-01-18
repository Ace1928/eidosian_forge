from .parseVertexGramMatrixFile import *
from .verificationError import *
from .orb import __path__ as orb_path
from snappy.snap.t3mlite import Mcomplex
import subprocess
import tempfile
import shutil
import os
def compute_approx_hyperbolic_structure_orb(triangulation, verbose=False):
    """
    Calls Orb to compute an approximate solution to the edge equation
    for the given snappy.Triangulation.

    The result is a veriClosed.HyperbolicStructure where the edge lengths
    are in SageMath's RealDoubleField.
    """
    with TmpDir() as tmp_dir:
        vgm_file_path = _compute_vertex_gram_matrix_file_orb(triangulation, tmp_dir.path, verbose=verbose)
        return compute_approx_hyperbolic_structure_from_vertex_gram_matrix_file(Mcomplex(triangulation), vgm_file_path)