from functools import reduce as pyreduce
def reduce_wrapper(seq, res=None, init=0):
    r = pyreduce(func, seq, init)
    if res is not None:
        res[0] = r
        return None
    else:
        return r