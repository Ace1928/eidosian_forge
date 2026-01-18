import os
import platform
def extract_bytes(mv):
    """Retrieve bytes out of the given input buffer.

    :param mv: input :py:func:`buffer`
    :type mv: memoryview or bytes

    :return: unwrapped bytes
    :rtype: bytes

    :raises ValueError: if the input is not one of \\
                        :py:class:`memoryview`/:py:func:`buffer` \\
                        or :py:class:`bytes`
    """
    if isinstance(mv, memoryview):
        return mv.tobytes()
    if isinstance(mv, bytes):
        return mv
    raise ValueError('extract_bytes() only accepts bytes and memoryview/buffer')