import re
def get_filetype_from_buffer(buf, max_lines=5):
    """
    Scan the buffer for modelines and return filetype if one is found.
    """
    lines = buf.splitlines()
    for l in lines[-1:-max_lines - 1:-1]:
        ret = get_filetype_from_line(l)
        if ret:
            return ret
    for i in range(max_lines, -1, -1):
        if i < len(lines):
            ret = get_filetype_from_line(lines[i])
            if ret:
                return ret
    return None