import re
from prompt_toolkit.key_binding import KeyPressEvent
def brackets(event: KeyPressEvent):
    """Auto-close brackets"""
    event.current_buffer.insert_text('[]')
    event.current_buffer.cursor_left()