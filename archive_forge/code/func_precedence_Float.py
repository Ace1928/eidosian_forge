def precedence_Float(item):
    if item < 0:
        return PRECEDENCE['Add']
    return PRECEDENCE['Atom']