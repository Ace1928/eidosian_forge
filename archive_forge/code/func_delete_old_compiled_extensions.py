import os
from _pydevd_bundle.pydevd_constants import USE_CYTHON_FLAG, ENV_TRUE_LOWER_VALUES, \
from _pydev_bundle import pydev_log
def delete_old_compiled_extensions():
    import _pydevd_bundle
    cython_extensions_dir = os.path.dirname(os.path.dirname(_pydevd_bundle.__file__))
    _pydevd_bundle_ext_dir = os.path.dirname(_pydevd_bundle.__file__)
    _pydevd_frame_eval_ext_dir = os.path.join(cython_extensions_dir, '_pydevd_frame_eval_ext')
    try:
        import shutil
        for file in os.listdir(_pydevd_bundle_ext_dir):
            if file.startswith('pydevd') and file.endswith('.so'):
                os.remove(os.path.join(_pydevd_bundle_ext_dir, file))
        for file in os.listdir(_pydevd_frame_eval_ext_dir):
            if file.startswith('pydevd') and file.endswith('.so'):
                os.remove(os.path.join(_pydevd_frame_eval_ext_dir, file))
        build_dir = os.path.join(cython_extensions_dir, 'build')
        if os.path.exists(build_dir):
            shutil.rmtree(os.path.join(cython_extensions_dir, 'build'))
    except OSError:
        pydev_log.error_once('warning: failed to delete old cython speedups. Please delete all *.so files from the directories "%s" and "%s"' % (_pydevd_bundle_ext_dir, _pydevd_frame_eval_ext_dir))