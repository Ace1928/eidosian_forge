import warnings
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import script_ops
from tensorflow.python.util import function_utils
from tensorflow.python.util import variable_utils
@property
def output_types(self):
    return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), self._output_structure)