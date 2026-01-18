import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def pre_and_post_stack_effect(self, jump=None):
    _effect = self.stack_effect(jump=jump)
    _opname = _opcode.opname[self._opcode]
    if _opname.startswith('DUP_TOP'):
        return (_effect * -1, _effect * 2)
    if _pushes_back(_opname):
        return (_effect - 1, 1)
    if _opname.startswith('UNPACK_'):
        return (-1, _effect + 1)
    if _opname == 'FOR_ITER' and (not jump):
        return (-1, 2)
    if _opname == 'ROT_N':
        return (-self._arg, self._arg)
    return {'ROT_TWO': (-2, 2), 'ROT_THREE': (-3, 3), 'ROT_FOUR': (-4, 4)}.get(_opname, (_effect, 0))