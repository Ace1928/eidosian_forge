import re
from prompt_toolkit.key_binding import KeyPressEvent
def docstring_double_quotes(event: KeyPressEvent):
    """Auto-close docstring (double quotes)"""
    event.current_buffer.insert_text('""""')
    event.current_buffer.cursor_left(3)