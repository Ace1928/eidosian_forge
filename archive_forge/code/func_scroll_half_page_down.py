from __future__ import unicode_literals
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from six.moves import range
def scroll_half_page_down(event):
    """
    Same as ControlF, but only scroll half a page.
    """
    scroll_forward(event, half=True)