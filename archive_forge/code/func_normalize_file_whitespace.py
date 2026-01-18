import re
def normalize_file_whitespace(text):
    """remove initial and final whitespace on each line, replace any interal
    whitespace with one space, and remove trailing blank lines"""
    lines_out = []
    for l in text.strip().splitlines():
        lines_out.append(re.sub('\\s+', ' ', l.strip()))
    return '\n'.join(lines_out)