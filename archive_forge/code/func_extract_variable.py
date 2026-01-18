from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def extract_variable(inference_state, path, module_node, name, pos, until_pos):
    nodes = _find_nodes(module_node, pos, until_pos)
    debug.dbg('Extracting nodes: %s', nodes)
    is_expression, message = _is_expression_with_error(nodes)
    if not is_expression:
        raise RefactoringError(message)
    generated_code = name + ' = ' + _expression_nodes_to_string(nodes)
    file_to_node_changes = {path: _replace(nodes, name, generated_code, pos)}
    return Refactoring(inference_state, file_to_node_changes)