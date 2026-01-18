from unittest import TestResult
import testresources
from testresources.tests import TestUtil
def stopResetResource(self, resource):
    self._calls.append(('reset', 'stop', resource))