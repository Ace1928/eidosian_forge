import sys
def cancel_patches_in_sys_module():
    sys.exc_info = sys.system_exc_info
    import builtins
    if hasattr(sys, 'builtin_orig_reload'):
        builtins.reload = sys.builtin_orig_reload
    if hasattr(sys, 'imp_orig_reload'):
        import imp
        imp.reload = sys.imp_orig_reload
    if hasattr(sys, 'importlib_orig_reload'):
        import importlib
        importlib.reload = sys.importlib_orig_reload
    del builtins