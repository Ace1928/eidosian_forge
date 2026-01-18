import re
import copy
import inspect
import ast
import textwrap
def expand_function_ast_varargs(fn_ast, expand_number):
    """
    Given a function AST that use a variable length positional argument
    (e.g. *args), return a function that replaces the use of this argument
    with one or more fixed arguments.

    To be supported, a function must have a starred argument in the function
    signature, and it may only use this argument in starred form as the
    input to other functions.

    For example, suppose expand_number is 3 and fn_ast is an AST
    representing this function...

    def my_fn1(a, b, *args):
        print(a, b)
        other_fn(a, b, *args)

    Then this function will return the AST of a function equivalent to...

    def my_fn1(a, b, _0, _1, _2):
        print(a, b)
        other_fn(a, b, _0, _1, _2)

    If the input function uses `args` for anything other than passing it to
    other functions in starred form, an error will be raised.

    Parameters
    ----------
    fn_ast: ast.FunctionDef
    expand_number: int

    Returns
    -------
    ast.FunctionDef
    """
    assert isinstance(fn_ast, ast.Module)
    fn_ast = copy.deepcopy(fn_ast)
    fndef_ast = fn_ast.body[0]
    assert isinstance(fndef_ast, ast.FunctionDef)
    fn_args = fndef_ast.args
    fn_vararg = fn_args.vararg
    if not fn_vararg:
        raise ValueError('Input function AST does not have a variable length positional argument\n(e.g. *args) in the function signature')
    assert fn_vararg
    if isinstance(fn_vararg, str):
        vararg_name = fn_vararg
    else:
        vararg_name = fn_vararg.arg
    before_name_visitor = NameVisitor()
    before_name_visitor.visit(fn_ast)
    expand_names = before_name_visitor.get_new_names(expand_number)
    expand_transformer = ExpandVarargTransformerStarred
    new_fn_ast = expand_transformer(vararg_name, expand_names).visit(fn_ast)
    new_fndef_ast = new_fn_ast.body[0]
    new_fndef_ast.args.args.extend([_build_arg(name=name) for name in expand_names])
    new_fndef_ast.args.vararg = None
    after_name_visitor = NameVisitor()
    after_name_visitor.visit(new_fn_ast)
    if vararg_name in after_name_visitor.names:
        raise ValueError('The variable length positional argument {n} is used in an unsupported context\n'.format(n=vararg_name))
    fndef_ast.decorator_list = []
    ast.fix_missing_locations(new_fn_ast)
    return new_fn_ast