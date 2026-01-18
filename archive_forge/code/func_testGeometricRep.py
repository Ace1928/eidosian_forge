from snappy import Manifold, pari, ptolemy
from snappy.ptolemy import solutions_from_magma, Flattenings, parse_solutions
from snappy.ptolemy.processFileBase import get_manifold
from snappy.ptolemy import __path__ as ptolemy_paths
from snappy.ptolemy.coordinates import PtolemyCannotBeCheckedError
from snappy.sage_helper import _within_sage, doctest_modules
from snappy.pari import pari
import bz2
import os
import sys
def testGeometricRep(compute_solutions):
    from snappy.ptolemy import geometricRep
    M = Manifold('m019')
    if compute_solutions:
        sol = geometricRep.compute_geometric_solution(M)
    else:
        from urllib.request import pathname2url
        url = pathname2url(os.path.abspath(testing_files_directory))
        sol = geometricRep.retrieve_geometric_solution(M, data_url=url)
    sol['c_0011_2']
    assert any([abs(vol - 2.9441064867) < 1e-09 for vol in sol.volume_numerical()])