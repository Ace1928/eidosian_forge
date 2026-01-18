import keyword
import os
import re
import subprocess
import sys
from taskflow import test
def make_output_files():
    """Generate output files for all examples."""
    for example_name, _safe_name in iter_examples():
        print('Running %s' % example_name)
        print('Please wait...')
        output = run_example(example_name)
        with open(expected_output_path(example_name), 'w') as f:
            f.write(output)