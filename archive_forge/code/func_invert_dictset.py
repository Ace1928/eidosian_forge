def invert_dictset(d):
    """Invert a dict with keys matching a set of values, turned into lists."""
    result = {}
    for k, c in d.items():
        for v in c:
            keys = result.setdefault(v, [])
            keys.append(k)
    return result