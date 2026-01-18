import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def getme(branch):
    return MyLogFormatter