from .error import VerificationError
def as_c_expr(self):
    if self.op is None:
        assert isinstance(self.arg, str)
        return '(_cffi_opcode_t)(%s)' % (self.arg,)
    classname = CLASS_NAME[self.op]
    return '_CFFI_OP(_CFFI_OP_%s, %s)' % (classname, self.arg)