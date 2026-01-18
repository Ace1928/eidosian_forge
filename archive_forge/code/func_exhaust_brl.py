def exhaust_brl(expr):
    seen = {expr}
    for nexpr in brule(expr):
        if nexpr not in seen:
            seen.add(nexpr)
            yield from exhaust_brl(nexpr)
    if seen == {expr}:
        yield expr