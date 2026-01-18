def onaction(brule, fn):

    def onaction_brl(expr):
        for result in brule(expr):
            if result != expr:
                fn(brule, expr, result)
            yield result
    return onaction_brl