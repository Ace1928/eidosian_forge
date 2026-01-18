from __future__ import nested_scopes
from _pydev_bundle._pydev_saved_modules import threading
import os
from _pydev_bundle import pydev_log
def patched_import(name, *args, **kwargs):
    if patch_qt_on_import == name or name.startswith(dotted):
        builtins.__import__ = original_import
        cancel_patches_in_sys_module()
        _internal_patch_qt(get_qt_core_module())
    return original_import(name, *args, **kwargs)