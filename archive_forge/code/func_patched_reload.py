import sys
def patched_reload(orig_reload):

    def pydev_debugger_reload(module):
        orig_reload(module)
        if module.__name__ == 'sys':
            patch_sys_module()
    return pydev_debugger_reload