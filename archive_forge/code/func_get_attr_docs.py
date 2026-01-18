import ast
import inspect
import sys
import textwrap
import typing as T
from types import ModuleType
from .common import Docstring, DocstringParam
def get_attr_docs(self, component: T.Any) -> T.Dict[str, T.Tuple[str, T.Optional[str], T.Optional[str]]]:
    """Get attribute docstrings from the given component.

        :param component: component to process (class or module)
        :returns: for each attribute docstring, a tuple with (description,
            type, default)
        """
    self.attr_docs = {}
    self.prev_attr = None
    try:
        source = textwrap.dedent(inspect.getsource(component))
    except OSError:
        pass
    else:
        tree = ast.parse(source)
        if inspect.ismodule(component):
            self.visit(tree)
        elif isinstance(tree, ast.Module) and isinstance(tree.body[0], ast.ClassDef):
            self.visit(tree.body[0])
    return self.attr_docs