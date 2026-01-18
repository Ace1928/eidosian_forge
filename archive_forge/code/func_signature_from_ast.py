from inspect import Parameter, Signature, getsource
from typing import Any, Dict, List, cast
import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import inspect, logging
def signature_from_ast(node: ast.FunctionDef, bound_method: bool, type_comment: ast.FunctionDef) -> Signature:
    """Return a Signature object for the given *node*.

    :param bound_method: Specify *node* is a bound method or not
    """
    params = []
    if hasattr(node.args, 'posonlyargs'):
        for arg in node.args.posonlyargs:
            param = Parameter(arg.arg, Parameter.POSITIONAL_ONLY, annotation=arg.type_comment)
            params.append(param)
    for arg in node.args.args:
        param = Parameter(arg.arg, Parameter.POSITIONAL_OR_KEYWORD, annotation=arg.type_comment or Parameter.empty)
        params.append(param)
    if node.args.vararg:
        param = Parameter(node.args.vararg.arg, Parameter.VAR_POSITIONAL, annotation=node.args.vararg.type_comment or Parameter.empty)
        params.append(param)
    for arg in node.args.kwonlyargs:
        param = Parameter(arg.arg, Parameter.KEYWORD_ONLY, annotation=arg.type_comment or Parameter.empty)
        params.append(param)
    if node.args.kwarg:
        param = Parameter(node.args.kwarg.arg, Parameter.VAR_KEYWORD, annotation=node.args.kwarg.type_comment or Parameter.empty)
        params.append(param)
    if bound_method and params:
        params.pop(0)
    if not_suppressed(type_comment.argtypes):
        for i, param in enumerate(params):
            params[i] = param.replace(annotation=type_comment.argtypes[i])
    if node.returns:
        return Signature(params, return_annotation=node.returns)
    elif type_comment.returns:
        return Signature(params, return_annotation=ast_unparse(type_comment.returns))
    else:
        return Signature(params)