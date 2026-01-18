import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, List, NamedTuple, Optional, Tuple
from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory
def normalize_source_lines(sourcelines: List[str]) -> List[str]:
    """
    This helper function accepts a list of source lines. It finds the
    indentation level of the function definition (`def`), then it indents
    all lines in the function body to a point at or greater than that
    level. This allows for comments and continued string literals that
    are at a lower indentation than the rest of the code.
    Args:
        sourcelines: function source code, separated into lines by
                        the '
' character
    Returns:
        A list of source lines that have been correctly aligned
    """

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]
    idx = None
    for i, l in enumerate(sourcelines):
        if l.lstrip().startswith('def'):
            idx = i
            break
    if idx is None:
        return sourcelines
    fn_def = sourcelines[idx]
    whitespace = fn_def.split('def')[0]
    aligned_prefix = [whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]]
    aligned_suffix = [whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1:]]
    aligned_prefix.append(fn_def)
    return aligned_prefix + aligned_suffix