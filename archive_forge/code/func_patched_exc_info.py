import sys
def patched_exc_info(fun):

    def pydev_debugger_exc_info():
        type, value, traceback = fun()
        if type == ImportError:
            if traceback and hasattr(traceback, 'tb_next'):
                return (type, value, traceback.tb_next)
        return (type, value, traceback)
    return pydev_debugger_exc_info