from Cython.TestUtils import CythonTest
def test_cascaded_assignment(self):
    self.t(u'x = y = z = abc = 43')