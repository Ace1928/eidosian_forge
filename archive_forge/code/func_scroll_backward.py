from __future__ import unicode_literals
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from six.moves import range
def scroll_backward(event, half=False):
    """
    Scroll window up.
    """
    w = _current_window_for_event(event)
    b = event.cli.current_buffer
    if w and w.render_info:
        info = w.render_info
        scroll_height = info.window_height
        if half:
            scroll_height //= 2
        y = max(0, b.document.cursor_position_row - 1)
        height = 0
        while y > 0:
            line_height = info.get_height_for_line(y)
            if height + line_height < scroll_height:
                height += line_height
                y -= 1
            else:
                break
        b.cursor_position = b.document.translate_row_col_to_index(y, 0)