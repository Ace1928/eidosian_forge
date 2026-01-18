import re
from prompt_toolkit.key_binding import KeyPressEvent
def raw_string_parenthesis(event: KeyPressEvent):
    """Auto-close parenthesis in raw strings"""
    matches = re.match('.*(r|R)[\\"\'](-*)', event.current_buffer.document.current_line_before_cursor)
    dashes = matches.group(2) if matches else ''
    event.current_buffer.insert_text('()' + dashes)
    event.current_buffer.cursor_left(len(dashes) + 1)