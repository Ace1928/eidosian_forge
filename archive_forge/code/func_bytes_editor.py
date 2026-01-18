import datetime
from functools import partial
import logging
def bytes_editor(auto_set=True, enter_set=False, encoding=None):
    """ Factory function that returns a text editor for bytes.
    """
    from traitsui.api import TextEditor
    if encoding is None:
        format = bytes.hex
        evaluate = bytes.fromhex
    else:
        format = partial(bytes.decode, encoding=encoding)
        evaluate = partial(str.encode, encoding=encoding)
    return TextEditor(multi_line=True, format_func=format, evaluate=evaluate, auto_set=auto_set, enter_set=enter_set)