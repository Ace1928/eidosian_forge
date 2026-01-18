import gast as ast
import numpy as np
import numbers
def dtype_to_ast(name):
    if name in ('bool',):
        return ast.Attribute(ast.Name('builtins', ast.Load(), None, None), name, ast.Load())
    else:
        return ast.Attribute(ast.Name(mangle('numpy'), ast.Load(), None, None), name, ast.Load())