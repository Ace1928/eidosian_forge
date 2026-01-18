import re
import html
def handle_subscripts(s: str) -> str:
    s = replace_subscript(s, subscript=True)
    s = replace_subscript(s, subscript=False)
    return s