import sys
import getopt
import snappy
import snappy.snap.test
import spherogram.test
import snappy.matrix
import snappy.verify.test
import snappy.ptolemy.test
import snappy.raytracing.cohomology_fractal
import snappy.raytracing.geodesic_tube_info
import snappy.raytracing.geodesics
import snappy.raytracing.ideal_raytracing_data
import snappy.raytracing.upper_halfspace_utilities
import snappy.drilling
import snappy.exterior_to_link.test
import snappy.pari
from snappy.sage_helper import (_within_sage, doctest_modules, cyopengl_works,
from snappy import numeric_output_checker
def spherogram_doctester(verbose):
    ans = spherogram.test.run_doctests(verbose, print_info=False)
    snappy.number.Number._accuracy_for_testing = None
    if _within_sage:
        snappy.number.use_field_conversion('sage')
    return ans