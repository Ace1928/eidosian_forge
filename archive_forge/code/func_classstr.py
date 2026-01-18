def classstr(klass):
    if klass in classmap:
        return classmap[klass]
    else:
        return repr(klass)