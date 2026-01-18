import inspect
import wsme.api
def scan_api(controller, path=[], objects=[]):
    """
    Recursively iterate a controller api entries.
    """
    for name in dir(controller):
        if name.startswith('_'):
            continue
        a = getattr(controller, name)
        if a in objects:
            continue
        if inspect.ismethod(a):
            if wsme.api.iswsmefunction(a):
                yield (path + [name], a, [])
        elif inspect.isclass(a):
            continue
        else:
            if len(path) > APIPATH_MAXLEN:
                raise ValueError('Path is too long: ' + str(path))
            for i in scan_api(a, path + [name], objects + [a]):
                yield i