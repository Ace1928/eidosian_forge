from functools import reduce
def check_onnx_broadcast(dims1, dims2):
    broadcast = False
    supported = True
    len1 = len(dims1)
    len2 = len(dims2)
    numel1 = reduce(lambda x, y: x * y, dims1)
    numel2 = reduce(lambda x, y: x * y, dims2)
    if len1 < len2:
        broadcast = True
        if numel2 != 1:
            supported = False
    elif len1 > len2:
        broadcast = True
        if numel2 != 1 and dims1[len1 - len2:] != dims2:
            supported = False
    elif dims1 != dims2:
        broadcast = True
        if numel2 != 1:
            supported = False
    if not supported:
        raise ValueError(f'Numpy style broadcasting is not supported in ONNX. Input dims are: {dims1}, {dims2}')
    return broadcast