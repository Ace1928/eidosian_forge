import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def read_lines_sha256(lines):
    m = new_sha256()
    for l in lines:
        if isinstance(l, bytes):
            m.update(l)
        else:
            m.update(l.encode('UTF-8'))
    return m.hexdigest()