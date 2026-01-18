import io
import sys
import unittest
def getRunner(self):
    return unittest.TextTestRunner(resultclass=resultFactory, stream=io.StringIO())