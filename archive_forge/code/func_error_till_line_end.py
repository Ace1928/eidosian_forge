import re
from pygments.lexer import Lexer
from pygments.token import Text, Comment, Operator, Keyword, Name, Number, \
def error_till_line_end(self, start, text):
    """Mark everything from ``start`` to the end of the line as Error."""
    end = start
    try:
        while text[end] != '\n':
            end += 1
    except IndexError:
        end = len(text)
    if end != start:
        self.cur.append((start, Error, text[start:end]))
    end = self.whitespace(end, text)
    return end