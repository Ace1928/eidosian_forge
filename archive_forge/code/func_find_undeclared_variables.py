import typing as t
from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
def find_undeclared_variables(ast: nodes.Template) -> t.Set[str]:
    """Returns a set of all variables in the AST that will be looked up from
    the context at runtime.  Because at compile time it's not known which
    variables will be used depending on the path the execution takes at
    runtime, all variables are returned.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% set foo = 42 %}{{ bar + foo }}')
    >>> meta.find_undeclared_variables(ast) == {'bar'}
    True

    .. admonition:: Implementation

       Internally the code generator is used for finding undeclared variables.
       This is good to know because the code generator might raise a
       :exc:`TemplateAssertionError` during compilation and as a matter of
       fact this function can currently raise that exception as well.
    """
    codegen = TrackingCodeGenerator(ast.environment)
    codegen.visit(ast)
    return codegen.undeclared_identifiers