from .. import osutils
def encode_copy_instruction(offset, length):
    """Convert this offset into a control code and bytes."""
    copy_command = 128
    copy_bytes = [None]
    for copy_bit in (1, 2, 4, 8):
        base_byte = offset & 255
        if base_byte:
            copy_command |= copy_bit
            copy_bytes.append(bytes([base_byte]))
        offset >>= 8
    if length is None:
        raise ValueError('cannot supply a length of None')
    if length > 65536:
        raise ValueError("we don't emit copy records for lengths > 64KiB")
    if length == 0:
        raise ValueError('We cannot emit a copy of length 0')
    if length != 65536:
        for copy_bit in (16, 32):
            base_byte = length & 255
            if base_byte:
                copy_command |= copy_bit
                copy_bytes.append(bytes([base_byte]))
            length >>= 8
    copy_bytes[0] = bytes([copy_command])
    return b''.join(copy_bytes)