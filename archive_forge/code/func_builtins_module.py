import parso
from jedi.file_io import FileIO
from jedi import debug
from jedi import settings
from jedi.inference import imports
from jedi.inference import recursion
from jedi.inference.cache import inference_state_function_cache
from jedi.inference import helpers
from jedi.inference.names import TreeNameDefinition
from jedi.inference.base_value import ContextualizedNode, \
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.syntax_tree import infer_expr_stmt, \
from jedi.inference.imports import follow_error_node_imports_if_possible
from jedi.plugins import plugin_manager
@property
@inference_state_function_cache()
def builtins_module(self):
    module_name = 'builtins'
    builtins_module, = self.import_module((module_name,), sys_path=[])
    return builtins_module