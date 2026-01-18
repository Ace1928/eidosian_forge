import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def attributesAsDict(self, hasIterAttributes):
    return {attr.name: attr for attr in hasIterAttributes.iterAttributes()}