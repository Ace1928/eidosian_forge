import os
import sys
def make_mod(tmpdir, pre_run=None):
    c_file = os.path.join(tmpdir, module_name + source_extension)
    log.info('generating cffi module %r' % c_file)
    mkpath(tmpdir)
    if pre_run is not None:
        pre_run(ext, ffi)
    updated = recompiler.make_c_source(ffi, module_name, source, c_file)
    if not updated:
        log.info('already up-to-date')
    return c_file