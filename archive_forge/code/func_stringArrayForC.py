import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
def stringArrayForC(self, strings):
    """Create a ctypes pointer to char-pointer set"""
    from OpenGL import arrays
    result = (ctypes.c_char_p * len(strings))()
    for i, s in enumerate(strings):
        result[i] = ctypes.cast(arrays.GLcharARBArray.dataPointer(s), ctypes.c_char_p)
    return result