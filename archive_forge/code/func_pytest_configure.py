import os
import sys
import warnings
from importlib.metadata import entry_points
import pytest
import networkx
def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: mark test as slow to run')
    backend = config.getoption('--backend')
    if backend is None:
        backend = os.environ.get('NETWORKX_TEST_BACKEND')
    if backend:
        networkx.utils.backends._dispatch._automatic_backends = [backend]
        fallback_to_nx = config.getoption('--fallback-to-nx')
        if not fallback_to_nx:
            fallback_to_nx = os.environ.get('NETWORKX_FALLBACK_TO_NX')
        networkx.utils.backends._dispatch._fallback_to_nx = bool(fallback_to_nx)
    if sys.version_info < (3, 10):
        backends = (ep for ep in entry_points()['networkx.backends'] if ep.name == 'nx-loopback')
    else:
        backends = entry_points(name='nx-loopback', group='networkx.backends')
    if backends:
        networkx.utils.backends.backends['nx-loopback'] = next(iter(backends))
    else:
        warnings.warn('\n\n             WARNING: Mixed NetworkX configuration! \n\n        This environment has mixed configuration for networkx.\n        The test object nx-loopback is not configured correctly.\n        You should not be seeing this message.\n        Try `pip install -e .`, or change your PYTHONPATH\n        Make sure python finds the networkx repo you are testing\n\n')