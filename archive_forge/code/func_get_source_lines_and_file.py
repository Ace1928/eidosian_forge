import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, List, NamedTuple, Optional, Tuple
from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory
def get_source_lines_and_file(obj: Any, error_msg: Optional[str]=None) -> Tuple[List[str], int, Optional[str]]:
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    """
    filename = None
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        msg = f"Can't get source for {obj}. TorchScript requires source access in order to carry out compilation, make sure original .py files are available."
        if error_msg:
            msg += '\n' + error_msg
        raise OSError(msg) from e
    return (sourcelines, file_lineno, filename)