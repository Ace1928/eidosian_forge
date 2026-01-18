from .error import VerificationError
def as_python_bytes(self):
    if self.op is None and self.arg.isdigit():
        value = int(self.arg)
        if value >= 2 ** 31:
            raise OverflowError('cannot emit %r: limited to 2**31-1' % (self.arg,))
        return format_four_bytes(value)
    if isinstance(self.arg, str):
        raise VerificationError('cannot emit to Python: %r' % (self.arg,))
    return format_four_bytes(self.arg << 8 | self.op)