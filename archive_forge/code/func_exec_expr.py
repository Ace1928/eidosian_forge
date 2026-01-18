from io import StringIO
import traceback
import threading
import pdb
import sys
def exec_expr(self, s):
    out = StringIO()
    exec_lock.acquire()
    save_stdout = sys.stdout
    try:
        debugger = _OutputRedirectingPdb(save_stdout)
        debugger.reset()
        pdb.set_trace = debugger.set_trace
        sys.stdout = out
        try:
            code = compile(s, '<web>', 'single', 0, 1)
            exec(code, self.globs, self.namespace)
            debugger.set_continue()
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc(file=out)
            debugger.set_continue()
    finally:
        sys.stdout = save_stdout
        exec_lock.release()
    return out.getvalue()