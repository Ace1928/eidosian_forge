import taskflow.engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
def run_fail(self, *args, **kwargs):
    self.values.append('fail')
    raise RuntimeError('Woot!')