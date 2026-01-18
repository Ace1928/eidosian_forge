import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def loadModuleAsDict(self, module):
    module.load()
    return self.attributesAsDict(module)