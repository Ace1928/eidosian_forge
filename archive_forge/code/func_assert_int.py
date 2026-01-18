from rsa._compat import is_integer
def assert_int(var, name):
    if is_integer(var):
        return
    raise TypeError('%s should be an integer, not %s' % (name, var.__class__))