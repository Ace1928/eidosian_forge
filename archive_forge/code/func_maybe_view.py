from functools import reduce
def maybe_view(tensor, size, check_same_size=True):
    if check_same_size and tensor.size() == size:
        return tensor
    return tensor.contiguous().view(size)