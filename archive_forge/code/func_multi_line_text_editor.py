import datetime
from functools import partial
import logging
def multi_line_text_editor(auto_set=True, enter_set=False):
    """ Factory function that returns a text editor for multi-line strings.
    """
    from traitsui.api import TextEditor
    return TextEditor(multi_line=True, auto_set=auto_set, enter_set=enter_set)