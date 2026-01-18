import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def dumpsource(object, alias='', new=False, enclose=True):
    """'dump to source', where the code includes a pickled object.

    If new=True and object is a class instance, then create a new
    instance using the unpacked class source code. If enclose, then
    create the object inside a function enclosure (thus minimizing
    any global namespace pollution).
    """
    from dill import dumps
    pik = repr(dumps(object))
    code = 'import dill\n'
    if enclose:
        stub = '__this_is_a_stub_variable__'
        pre = '%s = ' % stub
        new = False
    else:
        stub = alias
        pre = '%s = ' % stub if alias else alias
    if not new or not _isinstance(object):
        code += pre + 'dill.loads(%s)\n' % pik
    else:
        code += getsource(object.__class__, alias='', lstrip=True, force=True)
        mod = repr(object.__module__)
        code += pre + 'dill.loads(%s.replace(b%s,bytes(__name__,"UTF-8")))\n' % (pik, mod)
    if enclose:
        dummy = '__this_is_a_big_dummy_object__'
        dummy = _enclose(dummy, alias=alias)
        dummy = dummy.split('\n')
        code = dummy[0] + '\n' + indent(code) + '\n'.join(dummy[-3:])
    return code