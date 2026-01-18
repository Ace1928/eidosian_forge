import atexit
import errno
import importlib
import os
import sys
import tempfile
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
def load_ast(nodes, indentation='  ', include_source_map=False, delete_on_exit=True):
    """Loads the given AST as a Python module.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    nodes: Union[ast.AST, Iterable[ast.AST]], the code to compile, as an AST
      object.
    indentation: Text, the string to use for indentation.
    include_source_map: bool, whether return a source map.
    delete_on_exit: bool, whether to delete the temporary file used for
      compilation on exit.

  Returns:
    Tuple[module, Text, Dict[LineLocation, OriginInfo]], containing:
    the module containing the unparsed nodes, the source code corresponding to
    nodes, and the source map. Is include_source_map is False, the source map
    will be None.
  """
    if not isinstance(nodes, (list, tuple)):
        nodes = (nodes,)
    source = parser.unparse(nodes, indentation=indentation)
    module, _ = load_source(source, delete_on_exit)
    if include_source_map:
        source_map = origin_info.create_source_map(nodes, source, module.__file__)
    else:
        source_map = None
    return (module, source, source_map)