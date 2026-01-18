from snappy import verify, Manifold
from snappy.verify import upper_halfspace, cusp_shapes, cusp_areas, volume
from snappy.sage_helper import _within_sage, doctest_modules
import sys
import getopt
def generate_test_with_shapes_engine(module, engine):

    def result(verbose):
        globs = {'Manifold': Manifold}
        original = verify.CertifiedShapesEngine
        verify.CertifiedShapesEngine = engine
        r = doctest_modules([module], extraglobs=globs, verbose=verbose)
        verify.CertifiedShapesEngine = original
        return r
    result.__name__ = module.__name__ + '__with__' + engine.__name__
    return result