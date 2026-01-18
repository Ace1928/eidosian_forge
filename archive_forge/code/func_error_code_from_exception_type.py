import traceback
import warnings
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import _pywrap_py_exception_registry
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['errors.error_code_from_exception_type'])
def error_code_from_exception_type(cls):
    try:
        return _EXCEPTION_CLASS_TO_CODE[cls]
    except KeyError:
        warnings.warn('Unknown class exception')
        return UnknownError(None, None, 'Unknown class exception', None)