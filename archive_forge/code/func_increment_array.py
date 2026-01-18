import os
import pytest
from nipype.interfaces import utility
import nipype.pipeline.engine as pe
def increment_array(in_array):
    return in_array + 1