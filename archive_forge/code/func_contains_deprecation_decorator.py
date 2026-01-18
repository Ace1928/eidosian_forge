import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
def contains_deprecation_decorator(decorators):
    return any((d.decorator_name == 'deprecated' for d in decorators))