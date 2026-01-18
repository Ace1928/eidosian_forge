import os
import sys
from setuptools import setup
def process_args():
    extension_folder = None
    target_pydevd_name = None
    target_frame_eval = None
    force_cython = False
    for i, arg in enumerate(sys.argv[:]):
        if arg == '--build-lib':
            extension_folder = sys.argv[i + 1]
        if arg.startswith('--target-pyd-name='):
            sys.argv.remove(arg)
            target_pydevd_name = arg[len('--target-pyd-name='):]
        if arg.startswith('--target-pyd-frame-eval='):
            sys.argv.remove(arg)
            target_frame_eval = arg[len('--target-pyd-frame-eval='):]
        if arg == '--force-cython':
            sys.argv.remove(arg)
            force_cython = True
    return (extension_folder, target_pydevd_name, target_frame_eval, force_cython)