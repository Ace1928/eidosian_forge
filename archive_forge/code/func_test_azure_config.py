import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
@unittest.skipUnless(has_pyyaml, 'Requires pyyaml')
def test_azure_config(self):
    from yaml import Loader
    base_path = os.path.dirname(os.path.abspath(__file__))
    azure_pipe = os.path.join(base_path, '..', '..', 'azure-pipelines.yml')
    if not os.path.isfile(azure_pipe):
        self.skipTest("'azure-pipelines.yml' is not available")
    with open(os.path.abspath(azure_pipe), 'rt') as f:
        data = f.read()
    pipe_yml = yaml.load(data, Loader=Loader)
    templates = pipe_yml['jobs']
    start_indexes = []
    for tmplt in templates[:2]:
        matrix = tmplt['parameters']['matrix']
        for setup in matrix.values():
            start_indexes.append(setup['TEST_START_INDEX'])
    winpath = ['..', '..', 'buildscripts', 'azure', 'azure-windows.yml']
    azure_windows = os.path.join(base_path, *winpath)
    if not os.path.isfile(azure_windows):
        self.skipTest("'azure-windows.yml' is not available")
    with open(os.path.abspath(azure_windows), 'rt') as f:
        data = f.read()
    windows_yml = yaml.load(data, Loader=Loader)
    matrix = windows_yml['jobs'][0]['strategy']['matrix']
    for setup in matrix.values():
        start_indexes.append(setup['TEST_START_INDEX'])
    self.assertEqual(len(start_indexes), len(set(start_indexes)))
    lim_start_index = max(start_indexes) + 1
    expected = [*range(lim_start_index)]
    self.assertEqual(sorted(start_indexes), expected)
    self.assertEqual(lim_start_index, pipe_yml['variables']['TEST_COUNT'])