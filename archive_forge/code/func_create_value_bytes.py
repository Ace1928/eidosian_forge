from ctypes import Structure, c_int, c_byte
def create_value_bytes(value):
    if type(value) is bytes:
        v = bytearray(b'b')
        v = v + bytearray(value)
    elif type(value) is int:
        v = bytearray(b'i')
        v = v + bytearray(value.to_bytes(8, byteorder='big'))
    else:
        raise ValueError(f'invalid type for self.value {value}')
    return v