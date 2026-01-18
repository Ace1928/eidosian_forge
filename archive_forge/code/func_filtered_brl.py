def filtered_brl(expr):
    yield from filter(pred, brule(expr))