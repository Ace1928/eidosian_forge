import os
import pytest
from nipype.interfaces import utility
import nipype.pipeline.engine as pe
def make_random_array(size):
    return np.random.randn(size, size)