def sfilter(pred, brule):
    """ Yield only those results which satisfy the predicate """

    def filtered_brl(expr):
        yield from filter(pred, brule(expr))
    return filtered_brl