def alt(c):
    if type(c) == type(''):
        return chr(_ctoi(c) | 128)
    else:
        return _ctoi(c) | 128