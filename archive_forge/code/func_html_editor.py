import datetime
from functools import partial
import logging
def html_editor():
    """ Factory function for an "editor" that displays a multi-line string as
    interpreted HTML.
    """
    from traitsui.api import HTMLEditor
    return HTMLEditor()