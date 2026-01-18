import datetime
from functools import partial
import logging
def shell_editor():
    """ Factory function that returns a Python shell for editing Python values.
    """
    from traitsui.api import ShellEditor
    return ShellEditor()