from llvmlite.ir import Constant, IRBuilder
import llvmlite.ir
from numba.core import types, config, cgutils
def emit_cleanup(self):
    self.api.restore_thread(self.thread_state)
    self.argman.emit_cleanup()