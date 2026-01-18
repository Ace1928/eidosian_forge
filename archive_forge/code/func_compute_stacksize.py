import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def compute_stacksize(self, *, check_pre_and_post=True):
    cfg = _bytecode.ControlFlowGraph.from_bytecode(self)
    return cfg.compute_stacksize(check_pre_and_post=check_pre_and_post)