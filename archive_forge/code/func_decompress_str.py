from typing import Union
def decompress_str(data: Union[bytes, bytearray, memoryview], *, max_order: int=6, mem_size: int=16 << 20, encoding: str='UTF-8', variant: str='I') -> Union[bytes, str]:
    """Decompress a PPMd data, return a bytes object.

    Arguments
    data:      A bytes-like object, compressed data.
    max_order: An integer object represent max order of PPMd.
    mem_size:  An integer object represent memory size to use.
    encoding:  Encoding of compressed text data, when it is None return as bytes. Default is UTF-8
    variant:   A variant name of PPMd compression algorithms, accept only "H" or "I"
    """
    if not _is_bytelike(data):
        raise ValueError('Argument data should be bytes-like object.')
    if variant not in ['H', 'I', 'h', 'i']:
        raise ValueError('Unsupported PPMd variant')
    if variant in ['I', 'i']:
        return _decompress8(data, max_order, mem_size).decode(encoding)
    else:
        return _decompress7(data, max_order, mem_size).decode(encoding)