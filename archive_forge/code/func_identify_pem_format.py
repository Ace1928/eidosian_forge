from __future__ import absolute_import, division, print_function
def identify_pem_format(content, encoding='utf-8'):
    """Given the contents of a binary file, tests whether this could be a PEM file."""
    try:
        first_pem = extract_first_pem(content.decode(encoding))
        if first_pem is None:
            return False
        lines = first_pem.splitlines(False)
        if lines[0].startswith(PEM_START) and lines[0].endswith(PEM_END) and (len(lines[0]) > len(PEM_START) + len(PEM_END)):
            return True
    except UnicodeDecodeError:
        pass
    return False