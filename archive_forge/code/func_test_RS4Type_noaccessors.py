import pytest
import sys
import textwrap
import rpy2.robjects as robjects
import rpy2.robjects.methods as methods
def test_RS4Type_noaccessors():
    robjects.r['setClass']('Foo', robjects.r('list(foo="numeric")'))
    classdef = "\n    from rpy2 import robjects\n    from rpy2.robjects import methods\n    class Foo(methods.RS4, metaclass=methods.RS4_Type):\n        def __init__(self):\n            obj = robjects.r['new']('Foo')\n            super().__init__(obj)\n    "
    code = compile(textwrap.dedent(classdef), '<string>', 'exec')
    ns = dict()
    exec(code, ns)
    f = ns['Foo']()