from Cython.TestUtils import CythonTest
from Cython.Compiler.TreeFragment import *
from Cython.Compiler.Nodes import *
from Cython.Compiler.UtilNodes import *
def test_copy_is_taken(self):
    F = self.fragment(u'if True: x = 4')
    T1 = F.root
    T2 = F.copy()
    self.assertEqual('x', T2.stats[0].if_clauses[0].body.lhs.name)
    T2.stats[0].if_clauses[0].body.lhs.name = 'other'
    self.assertEqual('x', T1.stats[0].if_clauses[0].body.lhs.name)