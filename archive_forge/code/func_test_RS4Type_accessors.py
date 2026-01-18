import pytest
import sys
import textwrap
import rpy2.robjects as robjects
import rpy2.robjects.methods as methods
def test_RS4Type_accessors():
    robjects.r['setClass']('R_A', robjects.r('list(foo="numeric")'))
    robjects.r['setMethod']('length', signature='R_A', definition=robjects.r('function(x) 123'))
    classdef = "\n    from rpy2 import robjects\n    from rpy2.robjects import methods\n    class R_A(methods.RS4, metaclass=methods.RS4_Type):\n        __accessors__ = (\n            ('length', None,\n             'get_length', False, 'get the length'),\n            ('length', None,\n             None, True, 'length'))\n        def __init__(self):\n            obj = robjects.r['new']('R_A')\n            super().__init__(obj)            \n    "
    code = compile(textwrap.dedent(classdef), '<string>', 'exec')
    ns = dict()
    exec(code, ns)
    R_A = ns['R_A']

    class A(R_A):
        __rname__ = 'R_A'
    ra = R_A()
    assert ra.get_length()[0] == 123
    assert ra.length[0] == 123
    a = A()
    assert a.get_length()[0] == 123
    assert a.length[0] == 123