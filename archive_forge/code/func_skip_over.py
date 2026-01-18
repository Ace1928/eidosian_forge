import re
from prompt_toolkit.key_binding import KeyPressEvent
def skip_over(event: KeyPressEvent):
    """Skip over automatically added parenthesis/quote.

    (rather than adding another parenthesis/quote)"""
    event.current_buffer.cursor_right()