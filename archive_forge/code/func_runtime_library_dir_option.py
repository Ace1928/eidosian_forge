from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler
def runtime_library_dir_option(self, dir):
    return '-R%s' % dir