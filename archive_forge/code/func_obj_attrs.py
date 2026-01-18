import base64
import inspect
import builtins
def obj_attrs(msg_):
    """similar to obj_python_attrs() but deals with python reserved keywords
    """
    if isinstance(msg_, StringifyMixin):
        itr = msg_.stringify_attrs()
    else:
        itr = obj_python_attrs(msg_)
    for k, v in itr:
        if k.endswith('_') and k[:-1] in _RESERVED_KEYWORD:
            assert isinstance(msg_, StringifyMixin)
            k = k[:-1]
        yield (k, v)