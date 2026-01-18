import os
import sys
import warnings
from importlib.metadata import entry_points
import pytest
import networkx
def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help='run slow tests')
    parser.addoption('--backend', action='store', default=None, help='Run tests with a backend by auto-converting nx graphs to backend graphs')
    parser.addoption('--fallback-to-nx', action='store_true', default=False, help="Run nx function if a backend doesn't implement a dispatchable function (use with --backend)")