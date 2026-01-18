import os
import subprocess
import sys
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython.testing.decorators import skip_win32
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
import IPython
@skip_win32
def test_nest_embed():
    """test that `IPython.embed()` is nestable"""
    import pexpect
    ipy_prompt = ']:'
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.expect(ipy_prompt)
    child.timeout = 5 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.sendline('import IPython')
    child.expect(ipy_prompt)
    child.sendline('ip0 = get_ipython()')
    child.sendline('IPython.embed()')
    try:
        prompted = -1
        while prompted != 0:
            prompted = child.expect([ipy_prompt, '\r\n'])
    except pexpect.TIMEOUT as e:
        print(e)
    child.sendline('embed1 = get_ipython()')
    child.expect(ipy_prompt)
    child.sendline("print('true' if embed1 is not ip0 else 'false')")
    assert child.expect(['true\r\n', 'false\r\n']) == 0
    child.expect(ipy_prompt)
    child.sendline("print('true' if IPython.get_ipython() is embed1 else 'false')")
    assert child.expect(['true\r\n', 'false\r\n']) == 0
    child.expect(ipy_prompt)
    child.sendline('IPython.embed()')
    try:
        prompted = -1
        while prompted != 0:
            prompted = child.expect([ipy_prompt, '\r\n'])
    except pexpect.TIMEOUT as e:
        print(e)
    child.sendline('embed2 = get_ipython()')
    child.expect(ipy_prompt)
    child.sendline("print('true' if embed2 is not embed1 else 'false')")
    assert child.expect(['true\r\n', 'false\r\n']) == 0
    child.expect(ipy_prompt)
    child.sendline("print('true' if embed2 is IPython.get_ipython() else 'false')")
    assert child.expect(['true\r\n', 'false\r\n']) == 0
    child.expect(ipy_prompt)
    child.sendline('exit')
    child.expect(ipy_prompt)
    child.sendline("print('true' if get_ipython() is embed1 else 'false')")
    assert child.expect(['true\r\n', 'false\r\n']) == 0
    child.expect(ipy_prompt)
    child.sendline("print('true' if IPython.get_ipython() is embed1 else 'false')")
    assert child.expect(['true\r\n', 'false\r\n']) == 0
    child.expect(ipy_prompt)
    child.sendline('exit')
    child.expect(ipy_prompt)
    child.sendline("print('true' if get_ipython() is ip0 else 'false')")
    assert child.expect(['true\r\n', 'false\r\n']) == 0
    child.expect(ipy_prompt)
    child.sendline("print('true' if IPython.get_ipython() is ip0 else 'false')")
    assert child.expect(['true\r\n', 'false\r\n']) == 0
    child.expect(ipy_prompt)
    child.sendline('exit')
    child.close()