import io
import struct
import typing
@staticmethod
def key56_to_key64(key: bytes) -> bytes:
    """Convert 7 byte key to 8 bytes.

        This takes in an a bytes string of 7 bytes and converts it to a bytes
        string of 8 bytes with the odd parity bit being set to every 8 bits,

        For example

        b"\x01\x02\x03\x04\x05\x06\x07"
        00000001 00000010 00000011 00000100 00000101 00000110 00000111

        is converted to

        b"\x01\x80\x80a@)\x19\x0e"
        00000001 10000000 10000000 01100001 01000000 00101001 00011001 00001110

        https://crypto.stackexchange.com/questions/15799/des-with-actual-7-byte-key

        Args:
            key: The 7-byte sized key

        Returns:
            bytes: The expanded 8-byte key.
        """
    if len(key) != 7:
        raise ValueError('DES 7-byte key is not 7 bytes in length, actual: %d' % len(key))
    new_key = b''
    for i in range(0, 8):
        if i == 0:
            new_value = struct.unpack('B', key[i:i + 1])[0]
        elif i == 7:
            new_value = struct.unpack('B', key[6:7])[0]
            new_value = new_value << 1 & 255
        else:
            new_value = struct.unpack('B', key[i - 1:i])[0]
            next_value = struct.unpack('B', key[i:i + 1])[0]
            new_value = new_value << 8 - i & 255 | next_value >> i
        new_value = new_value & ~(1 << 0)
        new_value = new_value | int(not DES.bit_count(new_value) & 1)
        new_key += struct.pack('B', new_value)
    return new_key