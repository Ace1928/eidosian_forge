from __future__ import unicode_literals
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from six.moves import range
def scroll_half_page_up(event):
    """
    Same as ControlB, but only scroll half a page.
    """
    scroll_backward(event, half=True)