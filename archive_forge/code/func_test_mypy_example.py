import os
import sys
from subprocess import PIPE, STDOUT, Popen
import pytest
import zmq
@pytest.mark.skipif(not os.path.exists(examples_dir), reason='only test from examples directory')
@pytest.mark.parametrize('example', examples)
def test_mypy_example(example):
    example_dir = os.path.join(examples_dir, example)
    run_mypy('--disallow-untyped-calls', example_dir)