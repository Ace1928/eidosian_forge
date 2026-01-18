import re
import copy
import inspect
import ast
import textwrap
def function_ast_to_function(fn_ast, stacklevel=1):
    assert isinstance(fn_ast, ast.Module)
    fndef_ast = fn_ast.body[0]
    assert isinstance(fndef_ast, ast.FunctionDef)
    code = compile_function_ast(fn_ast)
    current_frame = inspect.currentframe()
    eval_frame = current_frame
    for _ in range(stacklevel):
        eval_frame = eval_frame.f_back
    eval_locals = eval_frame.f_locals
    eval_globals = eval_frame.f_globals
    del current_frame
    scope = copy.copy(eval_globals)
    scope.update(eval_locals)
    eval(code, scope)
    return scope[fndef_ast.name]