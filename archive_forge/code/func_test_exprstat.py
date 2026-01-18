from Cython.TestUtils import CythonTest
from Cython.Compiler.TreeFragment import *
from Cython.Compiler.Nodes import *
from Cython.Compiler.UtilNodes import *
def test_exprstat(self):
    F = self.fragment(u'PASS')
    pass_stat = PassStatNode(pos=None)
    T = F.substitute({'PASS': pass_stat})
    self.assertTrue(isinstance(T.stats[0], PassStatNode), T)