import sys
def allexcept(*args):
    newcats = cats[:]
    for arg in args:
        newcats.remove(arg)
    return u''.join((globals()[cat] for cat in newcats))