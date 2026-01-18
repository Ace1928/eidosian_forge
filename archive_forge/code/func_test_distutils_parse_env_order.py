import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
import importlib.metadata
from distutils.errors import DistutilsError
from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils
def test_distutils_parse_env_order(monkeypatch):
    from numpy.distutils.system_info import _parse_env_order
    env = 'NPY_TESTS_DISTUTILS_PARSE_ENV_ORDER'
    base_order = list('abcdef')
    monkeypatch.setenv(env, 'b,i,e,f')
    order, unknown = _parse_env_order(base_order, env)
    assert len(order) == 3
    assert order == list('bef')
    assert len(unknown) == 1
    monkeypatch.setenv(env, '')
    order, unknown = _parse_env_order(base_order, env)
    assert len(order) == 0
    assert len(unknown) == 0
    for prefix in '^!':
        monkeypatch.setenv(env, f'{prefix}b,i,e')
        order, unknown = _parse_env_order(base_order, env)
        assert len(order) == 4
        assert order == list('acdf')
        assert len(unknown) == 1
    with pytest.raises(ValueError):
        monkeypatch.setenv(env, 'b,^e,i')
        _parse_env_order(base_order, env)
    with pytest.raises(ValueError):
        monkeypatch.setenv(env, '!b,^e,i')
        _parse_env_order(base_order, env)