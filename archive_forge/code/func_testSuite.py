from twisted.trial import runner, unittest
def testSuite():
    ts = runner.TestSuite()
    ts.name = 'testSuite'
    return ts