from numpy.testing import assert_
import numpy.distutils.fcompiler
def test_gfortran_version(self):
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu95')
    for vs, version in gfortran_version_strings:
        v = fc.version_match(vs)
        assert_(v == version, (vs, v))