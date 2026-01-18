import os.path as _os_path
import platform as _platform
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as _CXX11_ABI_FLAG
from tensorflow.python.framework.versions import CXX_VERSION as _CXX_VERSION
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as _MONOLITHIC_BUILD
from tensorflow.python.framework.versions import VERSION as _VERSION
from tensorflow.python.platform import build_info
from tensorflow.python.util.tf_export import tf_export
@tf_export('sysconfig.get_include')
def get_include():
    """Get the directory containing the TensorFlow C++ header files.

  Returns:
    The directory as string.
  """
    import tensorflow as tf
    return _os_path.join(_os_path.dirname(tf.__file__), 'include')