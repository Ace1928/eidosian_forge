def read_weave(f):
    from .weave import Weave
    w = Weave(getattr(f, 'name', None))
    _read_weave_v5(f, w)
    return w