import sys
import re
from joblib.testing import raises, check_subprocess_call
def test_check_subprocess_call_wrong_command():
    wrong_command = '_a_command_that_does_not_exist_'
    with raises(OSError):
        check_subprocess_call([wrong_command])