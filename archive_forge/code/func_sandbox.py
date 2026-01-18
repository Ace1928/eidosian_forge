import os
import sys
import pytest
import six
@pytest.fixture
def sandbox(testbed):
    """
    Enables parts of the GAE sandbox that are relevant.
    Inserts the stub module import hook which causes the usage of
    appengine-specific httplib, httplib2, socket, etc.
    """
    try:
        from google.appengine.tools.devappserver2.python import sandbox
    except ImportError:
        from google.appengine.tools.devappserver2.python.runtime import sandbox
    for name in list(sys.modules):
        if name in sandbox.dist27.MODULE_OVERRIDES:
            del sys.modules[name]
    sys.meta_path.insert(0, sandbox.StubModuleImportHook())
    sys.path_importer_cache = {}
    yield testbed
    sys.meta_path = [x for x in sys.meta_path if not isinstance(x, sandbox.StubModuleImportHook)]
    sys.path_importer_cache = {}
    for name in list(sys.modules):
        if name in sandbox.dist27.MODULE_OVERRIDES:
            del sys.modules[name]