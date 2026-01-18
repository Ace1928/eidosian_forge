import os.path as _os_path
import platform as _platform
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as _CXX11_ABI_FLAG
from tensorflow.python.framework.versions import CXX_VERSION as _CXX_VERSION
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as _MONOLITHIC_BUILD
from tensorflow.python.framework.versions import VERSION as _VERSION
from tensorflow.python.platform import build_info
from tensorflow.python.util.tf_export import tf_export
@tf_export('sysconfig.get_compile_flags')
def get_compile_flags():
    """Returns the compilation flags for compiling with TensorFlow.

  The returned list of arguments can be passed to the compiler for compiling
  against TensorFlow headers. The result is platform dependent.

  For example, on a typical Linux system with Python 3.7 the following command
  prints `['-I/usr/local/lib/python3.7/dist-packages/tensorflow/include',
  '-D_GLIBCXX_USE_CXX11_ABI=1', '-DEIGEN_MAX_ALIGN_BYTES=64']`

  >>> print(tf.sysconfig.get_compile_flags())

  Returns:
    A list of strings for the compiler flags.
  """
    flags = []
    flags.append('-I%s' % get_include())
    flags.append('-D_GLIBCXX_USE_CXX11_ABI=%d' % _CXX11_ABI_FLAG)
    cxx_version_flag = None
    if _CXX_VERSION == 201103:
        cxx_version_flag = '--std=c++11'
    elif _CXX_VERSION == 201402:
        cxx_version_flag = '--std=c++14'
    elif _CXX_VERSION == 201703:
        cxx_version_flag = '--std=c++17'
    elif _CXX_VERSION == 202002:
        cxx_version_flag = '--std=c++20'
    if cxx_version_flag:
        flags.append(cxx_version_flag)
    flags.append('-DEIGEN_MAX_ALIGN_BYTES=%d' % pywrap_tf_session.get_eigen_max_align_bytes())
    return flags