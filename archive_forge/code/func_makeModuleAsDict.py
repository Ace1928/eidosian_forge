import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def makeModuleAsDict(self, source, path, name):
    return self.loadModuleAsDict(self.makeModule(source, path, name))