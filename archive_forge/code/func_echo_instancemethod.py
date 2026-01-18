import inspect
import sys
def echo_instancemethod(klass, method, write=sys.stdout.write):
    """ Change an instancemethod so that calls to it are echoed.

    Replacing a classmethod is a little more tricky.
    See: http://www.python.org/doc/current/ref/types.html
    """
    mname = method_name(method)
    never_echo = ('__str__', '__repr__')
    if mname in never_echo:
        pass
    elif is_classmethod(method, klass):
        setattr(klass, mname, classmethod(echo(method.__func__, write)))
    else:
        setattr(klass, mname, echo(method, write))