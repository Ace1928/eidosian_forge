import re
from hacking import core
@core.flake8ext
def assert_equal_true_or_false(logical_line):
    """Check for assertEqual(True, A) or assertEqual(False, A) sentences

    O323
    """
    res = assert_equal_with_true_re.search(logical_line) or assert_equal_with_false_re.search(logical_line)
    if res:
        yield (0, 'O323: assertEqual(True, A) or assertEqual(False, A) sentences not allowed')