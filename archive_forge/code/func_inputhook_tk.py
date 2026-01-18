from pydev_ipython.inputhook import stdin_ready
def inputhook_tk():
    while app.dooneevent(TCL_DONT_WAIT) == 1:
        if stdin_ready():
            break
    return 0