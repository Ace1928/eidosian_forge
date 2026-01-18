import os
import sys
import warnings
from importlib.metadata import entry_points
import pytest
import networkx
def pytest_collection_modifyitems(config, items):
    networkx.utils.backends._dispatch._is_testing = True
    if (automatic_backends := networkx.utils.backends._dispatch._automatic_backends):
        backend = networkx.utils.backends.backends[automatic_backends[0]].load()
        if hasattr(backend, 'on_start_tests'):
            getattr(backend, 'on_start_tests')(items)
    if config.getoption('--runslow'):
        return
    skip_slow = pytest.mark.skip(reason='need --runslow option to run')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)