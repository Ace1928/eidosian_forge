import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
@tf_contextlib.contextmanager
def silence():
    """Temporarily silence deprecation warnings."""
    global _PRINT_DEPRECATION_WARNINGS
    print_deprecation_warnings = _PRINT_DEPRECATION_WARNINGS
    _PRINT_DEPRECATION_WARNINGS = False
    yield
    _PRINT_DEPRECATION_WARNINGS = print_deprecation_warnings