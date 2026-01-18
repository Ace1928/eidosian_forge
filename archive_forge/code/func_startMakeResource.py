from unittest import TestResult
import testresources
from testresources.tests import TestUtil
def startMakeResource(self, resource):
    self._calls.append(('make', 'start', resource))