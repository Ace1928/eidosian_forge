import ast
import importlib
import importlib.util
import os
import sys
import threading
import types
import warnings
def attach_stub(package_name: str, filename: str):
    """Attach lazily loaded submodules, functions from a type stub.

    This is a variant on ``attach`` that will parse a `.pyi` stub file to
    infer ``submodules`` and ``submod_attrs``. This allows static type checkers
    to find imports, while still providing lazy loading at runtime.

    Parameters
    ----------
    package_name : str
        Typically use ``__name__``.
    filename : str
        Path to `.py` file which has an adjacent `.pyi` file.
        Typically use ``__file__``.

    Returns
    -------
    __getattr__, __dir__, __all__
        The same output as ``attach``.

    Raises
    ------
    ValueError
        If a stub file is not found for `filename`, or if the stubfile is formmated
        incorrectly (e.g. if it contains an relative import from outside of the module)
    """
    stubfile = filename if filename.endswith('i') else f'{os.path.splitext(filename)[0]}.pyi'
    if not os.path.exists(stubfile):
        raise ValueError(f'Cannot load imports from non-existent stub {stubfile!r}')
    with open(stubfile) as f:
        stub_node = ast.parse(f.read())
    visitor = _StubVisitor()
    visitor.visit(stub_node)
    return attach(package_name, visitor._submodules, visitor._submod_attrs)