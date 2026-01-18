import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@classmethod
def dotted_completion(cls, line, sorted_keys, compositor_defs):
    """
        Supply the appropriate key in Store.options and supply
        suggestions for further completion.
        """
    completion_key, suggestions = (None, [])
    tokens = [t for t in reversed(line.replace('.', ' ').split())]
    for i, token in enumerate(tokens):
        key_checks = []
        if i >= 0:
            key_checks.append(tokens[i])
        if i >= 1:
            key_checks.append('.'.join([key_checks[-1], tokens[i - 1]]))
        if i >= 2:
            key_checks.append('.'.join([key_checks[-1], tokens[i - 2]]))
        for key in reversed(key_checks):
            if key in sorted_keys:
                completion_key = key
                depth = completion_key.count('.')
                suggestions = [k.split('.')[depth + 1] for k in sorted_keys if k.startswith(completion_key + '.')]
                return (completion_key, suggestions)
        if token in compositor_defs:
            completion_key = compositor_defs[token]
            break
    return (completion_key, suggestions)