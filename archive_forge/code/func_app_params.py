import subprocess
import sys
from collections import namedtuple
from io import StringIO
from subprocess import PIPE
from typing import Any, Callable, Dict, Generator, Optional, Tuple
import pytest
from sphinx.testing import util
from sphinx.testing.util import SphinxTestApp, SphinxTestAppWrapperForSkipBuilding
@pytest.fixture
def app_params(request: Any, test_params: Dict, shared_result: SharedResult, sphinx_test_tempdir: str, rootdir: str) -> Tuple[Dict, Dict]:
    """
    Parameters that are specified by 'pytest.mark.sphinx' for
    sphinx.application.Sphinx initialization
    """
    pargs = {}
    kwargs: Dict[str, Any] = {}
    for info in reversed(list(request.node.iter_markers('sphinx'))):
        for i, a in enumerate(info.args):
            pargs[i] = a
        kwargs.update(info.kwargs)
    args = [pargs[i] for i in sorted(pargs.keys())]
    if test_params['shared_result']:
        if 'srcdir' in kwargs:
            raise pytest.Exception('You can not specify shared_result and srcdir in same time.')
        kwargs['srcdir'] = test_params['shared_result']
        restore = shared_result.restore(test_params['shared_result'])
        kwargs.update(restore)
    testroot = kwargs.pop('testroot', 'root')
    kwargs['srcdir'] = srcdir = sphinx_test_tempdir / kwargs.get('srcdir', testroot)
    if rootdir and (not srcdir.exists()):
        testroot_path = rootdir / ('test-' + testroot)
        testroot_path.copytree(srcdir)
    return namedtuple('app_params', 'args,kwargs')(args, kwargs)