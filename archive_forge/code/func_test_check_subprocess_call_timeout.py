import sys
import re
from joblib.testing import raises, check_subprocess_call
def test_check_subprocess_call_timeout():
    code_timing_out = '\n'.join(['import time', 'import sys', 'print("before sleep on stdout")', 'sys.stdout.flush()', 'sys.stderr.write("before sleep on stderr")', 'sys.stderr.flush()', 'time.sleep(10)', 'print("process should have be killed before")', 'sys.stdout.flush()'])
    pattern = re.compile('Non-zero return code:.+Stdout:\nbefore sleep on stdout\\s+Stderr:\nbefore sleep on stderr', re.DOTALL)
    with raises(ValueError) as excinfo:
        check_subprocess_call([sys.executable, '-c', code_timing_out], timeout=1)
    excinfo.match(pattern)