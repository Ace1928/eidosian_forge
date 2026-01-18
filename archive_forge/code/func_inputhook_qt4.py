import os
import signal
import threading
from pydev_ipython.qt_for_kernel import QtCore, QtGui
from pydev_ipython.inputhook import allow_CTRL_C, ignore_CTRL_C, stdin_ready
def inputhook_qt4():
    """PyOS_InputHook python hook for Qt4.

        Process pending Qt events and if there's no pending keyboard
        input, spend a short slice of time (50ms) running the Qt event
        loop.

        As a Python ctypes callback can't raise an exception, we catch
        the KeyboardInterrupt and temporarily deactivate the hook,
        which will let a *second* CTRL+C be processed normally and go
        back to a clean prompt line.
        """
    try:
        allow_CTRL_C()
        app = QtCore.QCoreApplication.instance()
        if not app:
            return 0
        app.processEvents(QtCore.QEventLoop.AllEvents, 300)
        if not stdin_ready():
            timer = QtCore.QTimer()
            event_loop = QtCore.QEventLoop()
            timer.timeout.connect(event_loop.quit)
            while not stdin_ready():
                timer.start(50)
                event_loop.exec_()
                timer.stop()
    except KeyboardInterrupt:
        global got_kbdint, sigint_timer
        ignore_CTRL_C()
        got_kbdint = True
        mgr.clear_inputhook()
        if os.name == 'posix':
            pid = os.getpid()
            if not sigint_timer:
                sigint_timer = threading.Timer(0.01, os.kill, args=[pid, signal.SIGINT])
                sigint_timer.start()
        else:
            print('\nKeyboardInterrupt - Ctrl-C again for new prompt')
    except:
        ignore_CTRL_C()
        from traceback import print_exc
        print_exc()
        print('Got exception from inputhook_qt4, unregistering.')
        mgr.clear_inputhook()
    finally:
        allow_CTRL_C()
    return 0