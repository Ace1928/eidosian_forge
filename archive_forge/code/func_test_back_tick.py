import os
from os.path import dirname, pathsep
import pytest
from ..testers import PYTHON, back_tick, run_mod_cmd
def test_back_tick():
    cmd = f'''{PYTHON} -c "print('Hello')"'''
    assert back_tick(cmd) == 'Hello'
    assert back_tick(cmd, ret_err=True) == ('Hello', '')
    assert back_tick(cmd, True, False) == (b'Hello', b'')
    cmd = f'{PYTHON} -c "raise ValueError()"'
    with pytest.raises(RuntimeError):
        back_tick(cmd)