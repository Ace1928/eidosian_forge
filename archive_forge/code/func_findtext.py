import re
def findtext(elem, path, default=None, namespaces=None, with_prefixes=True):
    el = find(elem, path, namespaces, with_prefixes=with_prefixes)
    if el is None:
        return default
    else:
        return el.text or ''