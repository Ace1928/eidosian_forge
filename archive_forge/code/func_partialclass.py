from functools import partialmethod
from torch import optim
def partialclass(cls, *args, **kwargs):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return NewCls