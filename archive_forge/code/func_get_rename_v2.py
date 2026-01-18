import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
def get_rename_v2(name):
    if name not in all_renames_v2.symbol_renames:
        return None
    return all_renames_v2.symbol_renames[name]