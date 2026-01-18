import re
from prompt_toolkit.key_binding import KeyPressEvent
def braces(event: KeyPressEvent):
    """Auto-close braces"""
    event.current_buffer.insert_text('{}')
    event.current_buffer.cursor_left()