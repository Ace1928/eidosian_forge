import typing
def get_overloads_for(func: typing.Callable):
    return _overloads.get(_get_fullqual_name(func), [])