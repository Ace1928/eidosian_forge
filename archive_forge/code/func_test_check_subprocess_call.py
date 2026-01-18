import sys
import re
from joblib.testing import raises, check_subprocess_call
def test_check_subprocess_call():
    code = '\n'.join(['result = 1 + 2 * 3', 'print(result)', 'my_list = [1, 2, 3]', 'print(my_list)'])
    check_subprocess_call([sys.executable, '-c', code])
    check_subprocess_call([sys.executable, '-c', code], stdout_regex='7\\s{1,2}\\[1, 2, 3\\]')