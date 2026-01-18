from pecan.compat import getargspec as _getargspec
def iscontroller(obj):
    return getattr(obj, 'exposed', False)