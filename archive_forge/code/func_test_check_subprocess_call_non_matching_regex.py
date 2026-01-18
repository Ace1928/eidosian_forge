import sys
import re
from joblib.testing import raises, check_subprocess_call
def test_check_subprocess_call_non_matching_regex():
    code = '42'
    non_matching_pattern = '_no_way_this_matches_anything_'
    with raises(ValueError) as excinfo:
        check_subprocess_call([sys.executable, '-c', code], stdout_regex=non_matching_pattern)
    excinfo.match('Unexpected stdout.+{}'.format(non_matching_pattern))