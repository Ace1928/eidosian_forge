import io
import re
from contextlib import redirect_stdout
import pytest
from numpy.distutils import log
@pytest.mark.parametrize('func_name', ['error', 'warn', 'info', 'debug'])
def test_log_prefix(func_name):
    func = getattr(log, func_name)
    msg = f'{func_name} message'
    f = io.StringIO()
    with redirect_stdout(f):
        func(msg)
    out = f.getvalue()
    assert out
    clean_out = r_ansi.sub('', out)
    line = next((line for line in clean_out.splitlines()))
    assert line == f'{func_name.upper()}: {msg}'