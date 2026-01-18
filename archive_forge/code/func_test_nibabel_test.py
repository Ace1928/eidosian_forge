import pathlib
from unittest import mock
import pytest
import nibabel as nib
@pytest.mark.parametrize('verbose, v_args', [(-2, ['-qq']), (-1, ['-q']), (0, []), (1, ['-v']), (2, ['-vv'])])
@pytest.mark.parametrize('doctests', (True, False))
@pytest.mark.parametrize('coverage', (True, False))
def test_nibabel_test(verbose, v_args, doctests, coverage):
    expected_args = v_args + ['--doctest-modules', '--cov', 'nibabel', '--pyargs', 'nibabel']
    if not doctests:
        expected_args.remove('--doctest-modules')
    if not coverage:
        expected_args[-4:-2] = []
    with mock.patch('pytest.main') as pytest_main:
        nib.test(verbose=verbose, doctests=doctests, coverage=coverage)
    args, kwargs = pytest_main.call_args
    assert args == ()
    assert kwargs == {'args': expected_args}