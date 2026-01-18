import taskflow.engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
def revert_one(self, *args, **kwargs):
    self.values.append('revert one')