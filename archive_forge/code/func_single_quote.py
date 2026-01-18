import re
from prompt_toolkit.key_binding import KeyPressEvent
def single_quote(event: KeyPressEvent):
    """Auto-close single quotes"""
    event.current_buffer.insert_text("''")
    event.current_buffer.cursor_left()