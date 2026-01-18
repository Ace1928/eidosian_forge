import pathlib
from unittest import mock
import pytest
import nibabel as nib
def test_nibabel_bench():
    config_path = files('nibabel') / 'benchmarks/pytest.benchmark.ini'
    if not isinstance(config_path, pathlib.Path):
        raise unittest.SkipTest('Package is not unpacked; could get temp path')
    expected_args = ['-c', str(config_path), '--pyargs', 'nibabel']
    with mock.patch('pytest.main') as pytest_main:
        nib.bench(verbose=0)
    args, kwargs = pytest_main.call_args
    assert args == ()
    assert kwargs == {'args': expected_args}
    with mock.patch('pytest.main') as pytest_main:
        nib.bench(verbose=0, extra_argv=[])
    args, kwargs = pytest_main.call_args
    assert args == ()
    assert kwargs == {'args': expected_args}