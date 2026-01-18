from .python3_compat import iterkeys, iteritems, Mapping  #, u
def pre_unmunchify(obj):
    if isinstance(obj, Mapping):
        return dict()
    elif isinstance(obj, list):
        return type(obj)()
    elif isinstance(obj, tuple):
        type_factory = getattr(obj, '_make', type(obj))
        return type_factory((unmunchify_cycles(item) for item in obj))
    else:
        return obj