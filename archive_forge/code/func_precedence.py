def precedence(item):
    """Returns the precedence of a given object.

    This is the precedence for StrPrinter.
    """
    if hasattr(item, 'precedence'):
        return item.precedence
    try:
        mro = item.__class__.__mro__
    except AttributeError:
        return PRECEDENCE['Atom']
    for i in mro:
        n = i.__name__
        if n in PRECEDENCE_FUNCTIONS:
            return PRECEDENCE_FUNCTIONS[n](item)
        elif n in PRECEDENCE_VALUES:
            return PRECEDENCE_VALUES[n]
    return PRECEDENCE['Atom']