from __future__ import absolute_import, division, print_function
def split_pem_list(text, keep_inbetween=False):
    """
    Split concatenated PEM objects into a list of strings, where each is one PEM object.
    """
    result = []
    current = [] if keep_inbetween else None
    for line in text.splitlines(True):
        if line.strip():
            if not keep_inbetween and line.startswith('-----BEGIN '):
                current = []
            if current is not None:
                current.append(line)
                if line.startswith('-----END '):
                    result.append(''.join(current))
                    current = [] if keep_inbetween else None
    return result