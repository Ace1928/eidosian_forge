from unittest import TestResult
import testresources
from testresources.tests import TestUtil
def startResetResource(self, resource):
    self._calls.append(('reset', 'start', resource))