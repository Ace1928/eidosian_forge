from __future__ import (absolute_import, division, print_function)
import sys
def compile_source(path):
    """Compile the specified source file, printing an error if one occurs."""
    with open(path, 'rb') as source_fd:
        source = source_fd.read()
    try:
        compile(source, path, 'exec', dont_inherit=True)
    except SyntaxError as ex:
        extype, message, lineno, offset = (type(ex), ex.text, ex.lineno, ex.offset)
    except BaseException as ex:
        extype, message, lineno, offset = (type(ex), str(ex), 0, 0)
    else:
        return
    offset = offset or 0
    result = '%s:%d:%d: %s: %s' % (path, lineno, offset, extype.__name__, safe_message(message))
    if sys.version_info <= (3,):
        result = result.encode(ENCODING, ERRORS)
    print(result)