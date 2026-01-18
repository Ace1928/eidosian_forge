import re
def join_long_lines(text):
    """
    Deletes all backslash newline sequences. Inverse of break_long_lines.
    """
    return text.replace('\\\n', '')