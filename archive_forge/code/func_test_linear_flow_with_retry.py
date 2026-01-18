from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_linear_flow_with_retry(self):
    ret = retry.AlwaysRevert(requires=['a'], provides=['b'])
    f = lf.Flow('test', ret)
    self.assertIs(f.retry, ret)
    self.assertEqual('test_retry', ret.name)
    self.assertEqual(set(['a']), f.requires)
    self.assertEqual(set(['b']), f.provides)