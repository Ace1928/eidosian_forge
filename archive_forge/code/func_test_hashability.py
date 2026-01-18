from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_hashability(self):
    """
        In order for one test method to be runnable twice, two TestCase
        instances with the same test method name should not have the same
        hash value.
        """
    container = {}
    container[self.first] = None
    container[self.second] = None
    self.assertEqual(len(container), 2)