def ctrl(c):
    if type(c) == type(''):
        return chr(_ctoi(c) & 31)
    else:
        return _ctoi(c) & 31