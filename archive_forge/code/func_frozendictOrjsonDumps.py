def frozendictOrjsonDumps(obj, *args, **kwargs):
    if isinstance(obj, frozendict):
        obj = dict(obj)
    return oldOrjsonDumps(obj, *args, **kwargs)