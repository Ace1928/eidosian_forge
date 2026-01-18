import re
def iterfind(elem, path, namespaces=None, with_prefixes=True):
    selector = _build_path_iterator(path, namespaces, with_prefixes=with_prefixes)
    result = iter((elem,))
    for select in selector:
        result = select(result)
    return result