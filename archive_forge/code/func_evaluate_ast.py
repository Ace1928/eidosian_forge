import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict
def evaluate_ast(expression: ast.AST, state: Dict[str, Any], tools: Dict[str, Callable]):
    """
    Evaluate an absract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse trough the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abastract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignements.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
    """
    if isinstance(expression, ast.Assign):
        return evaluate_assign(expression, state, tools)
    elif isinstance(expression, ast.Call):
        return evaluate_call(expression, state, tools)
    elif isinstance(expression, ast.Constant):
        return expression.value
    elif isinstance(expression, ast.Dict):
        keys = [evaluate_ast(k, state, tools) for k in expression.keys]
        values = [evaluate_ast(v, state, tools) for v in expression.values]
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.For):
        return evaluate_for(expression, state, tools)
    elif isinstance(expression, ast.FormattedValue):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.If):
        return evaluate_if(expression, state, tools)
    elif hasattr(ast, 'Index') and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.JoinedStr):
        return ''.join([str(evaluate_ast(v, state, tools)) for v in expression.values])
    elif isinstance(expression, ast.List):
        return [evaluate_ast(elt, state, tools) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        return evaluate_name(expression, state, tools)
    elif isinstance(expression, ast.Subscript):
        return evaluate_subscript(expression, state, tools)
    else:
        raise InterpretorError(f'{expression.__class__.__name__} is not supported.')