from struct import unpack
def read_fixed(self, size):
    """Fixed instances are encoded using the number of bytes declared in the
        schema."""
    out = self.fo.read(size)
    if len(out) < size:
        raise EOFError(f'Expected {size} bytes, read {len(out)}')
    return out