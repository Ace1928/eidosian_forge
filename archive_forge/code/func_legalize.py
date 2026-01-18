import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def legalize(self):
    """Check that all the element of the list are valid and remove SetLineno."""
    lineno_pos = []
    set_lineno = None
    current_lineno = self.first_lineno
    for pos, instr in enumerate(self):
        if isinstance(instr, SetLineno):
            set_lineno = instr.lineno
            lineno_pos.append(pos)
            continue
        if not isinstance(instr, Instr):
            continue
        if set_lineno is not None:
            instr.lineno = set_lineno
        elif instr.lineno is None:
            instr.lineno = current_lineno
        else:
            current_lineno = instr.lineno
    for i in reversed(lineno_pos):
        del self[i]