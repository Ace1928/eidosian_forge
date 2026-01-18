import re
from string import Formatter
@staticmethod
def ansify(text):
    parser = AnsiParser()
    parser.feed(text.strip())
    tokens = parser.done(strict=False)
    return AnsiParser.colorize(tokens, None)