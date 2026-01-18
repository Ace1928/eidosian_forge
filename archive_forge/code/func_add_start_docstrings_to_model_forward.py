import functools
import re
import types
def add_start_docstrings_to_model_forward(*docstr):

    def docstring_decorator(fn):
        docstring = ''.join(docstr) + (fn.__doc__ if fn.__doc__ is not None else '')
        class_name = f'[`{fn.__qualname__.split('.')[0]}`]'
        intro = f'   The {class_name} forward method, overrides the `__call__` special method.'
        note = '\n\n    <Tip>\n\n    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]\n    instance afterwards instead of this since the former takes care of running the pre and post processing steps while\n    the latter silently ignores them.\n\n    </Tip>\n'
        fn.__doc__ = intro + note + docstring
        return fn
    return docstring_decorator