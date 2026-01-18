import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def do_one_token_transform(self, lines):
    """Find and run the transform earliest in the code.

        Returns (changed, lines).

        This method is called repeatedly until changed is False, indicating
        that all available transformations are complete.

        The tokens following IPython special syntax might not be valid, so
        the transformed code is retokenised every time to identify the next
        piece of special syntax. Hopefully long code cells are mostly valid
        Python, not using lots of IPython special syntax, so this shouldn't be
        a performance issue.
        """
    tokens_by_line = make_tokens_by_line(lines)
    candidates = []
    for transformer_cls in self.token_transformers:
        transformer = transformer_cls.find(tokens_by_line)
        if transformer:
            candidates.append(transformer)
    if not candidates:
        return (False, lines)
    ordered_transformers = sorted(candidates, key=TokenTransformBase.sortby)
    for transformer in ordered_transformers:
        try:
            return (True, transformer.transform(lines))
        except SyntaxError:
            pass
    return (False, lines)