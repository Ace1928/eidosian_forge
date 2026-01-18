import os
import sys
from tempfile import mkdtemp
from shutil import rmtree
import pytest
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
def dummyFunction(filename):
    """
        This function writes the value 45 to the given filename.
        """
    j = 0
    for i in range(0, 10):
        j += i
    with open(filename, 'w') as f:
        f.write(str(j))