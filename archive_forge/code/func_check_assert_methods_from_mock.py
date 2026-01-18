import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def check_assert_methods_from_mock(logical_line, filename):
    """Ensure that ``assert_*`` methods from ``mock`` library is used correctly

    N301 - base error number
    N302 - related to nonexistent "assert_called"
    N303 - related to nonexistent "assert_called_once"
    """
    correct_names = ['assert_any_call', 'assert_called_once_with', 'assert_called_with', 'assert_has_calls']
    ignored_files = ['./tests/unit/test_hacking.py']
    if filename.startswith('./tests') and filename not in ignored_files:
        pos, method_name, obj_name = _parse_assert_mock_str(logical_line)
        if pos:
            if method_name not in correct_names:
                error_number = 'N301'
                msg = "%(error_number)s:'%(method)s' is not present in `mock` library. %(custom_msg)s For more details, visit http://www.voidspace.org.uk/python/mock/ ."
                if method_name == 'assert_called':
                    error_number = 'N302'
                    custom_msg = "Maybe, you should try to use 'assertTrue(%s.called)' instead." % obj_name
                elif method_name == 'assert_called_once':
                    error_number = 'N303'
                    custom_msg = "Maybe, you should try to use 'assertEqual(1, %s.call_count)' or '%s.assert_called_once_with()' instead." % (obj_name, obj_name)
                else:
                    custom_msg = "Correct 'assert_*' methods: '%s'." % "', '".join(correct_names)
                yield (pos, msg % {'error_number': error_number, 'method': method_name, 'custom_msg': custom_msg})