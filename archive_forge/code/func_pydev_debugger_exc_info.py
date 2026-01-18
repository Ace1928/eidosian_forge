import sys
def pydev_debugger_exc_info():
    type, value, traceback = fun()
    if type == ImportError:
        if traceback and hasattr(traceback, 'tb_next'):
            return (type, value, traceback.tb_next)
    return (type, value, traceback)