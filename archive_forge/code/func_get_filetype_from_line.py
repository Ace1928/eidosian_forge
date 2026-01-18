import re
def get_filetype_from_line(l):
    m = modeline_re.search(l)
    if m:
        return m.group(1)