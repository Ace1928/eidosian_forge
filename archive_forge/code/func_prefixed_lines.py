import re
def prefixed_lines():
    for line in text.splitlines(True):
        yield (prefix + line if predicate(line) else line)