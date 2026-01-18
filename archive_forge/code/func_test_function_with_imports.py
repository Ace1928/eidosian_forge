import os
import pytest
from nipype.interfaces import utility
import nipype.pipeline.engine as pe
def test_function_with_imports(tmpdir):
    tmpdir.chdir()
    node = pe.Node(utility.Function(input_names=['size'], output_names=['random_array'], function=make_random_array, imports=['import numpy as np']), name='should_not_fail')
    print(node.inputs.function_str)
    node.inputs.size = 10
    node.run()