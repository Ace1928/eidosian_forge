import branchedDoubleCover
import veriClosed
from veriClosed import *
from veriClosed.verifyHyperbolicStructureEngine import *
from veriClosed.testing.cocycleTester import *
from veriClosed.testing import __path__ as testPaths
from snappy import Triangulation
from snappy.sage_helper import doctest_modules
import os
import sys
def testVerifyFromVertexGramMatrixFile():
    """
    Testing.
      
        >>> f = os.path.join(testPath, "m004_1_2.tri")
        >>> T = Triangulation(f, remove_finite_vertices = False)
        >>> bool(compute_verified_hyperbolic_structure(T, source = f + '.vgm'))
        True
    """