import os
import sys
def generate_mod(py_file):
    log.info('generating cffi module %r' % py_file)
    mkpath(os.path.dirname(py_file))
    updated = recompiler.make_py_source(ffi, module_name, py_file)
    if not updated:
        log.info('already up-to-date')