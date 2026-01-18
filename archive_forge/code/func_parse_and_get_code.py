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
def parse_and_get_code(self, code=None, path=None, use_latest_grammar=False, file_io=None, **kwargs):
    if code is None:
        if file_io is None:
            file_io = FileIO(path)
        code = file_io.read()
    code = parso.python_bytes_to_unicode(code, encoding='utf-8', errors='replace')
    if len(code) > settings._cropped_file_size:
        code = code[:settings._cropped_file_size]
    grammar = self.latest_grammar if use_latest_grammar else self.grammar
    return (grammar.parse(code=code, path=path, file_io=file_io, **kwargs), code)