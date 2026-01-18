import re
import numpy as np
def assert_lines_equal_ignoring_whitespace(test, expected_lines, actual_lines):
    """Assert equality in lines, ignoring all whitespace.

  Args:
    test: An instance of unittest.TestCase or its subtypes (e.g.,
      TensorFlowTestCase).
    expected_lines: Expected lines as an iterable of strings.
    actual_lines: Actual lines as an iterable of strings.
  """
    test.assertEqual(len(expected_lines), len(actual_lines), 'Mismatch in the number of lines: %d vs %d' % (len(expected_lines), len(actual_lines)))
    for expected_line, actual_line in zip(expected_lines, actual_lines):
        test.assertEqual(''.join(expected_line.split()), ''.join(actual_line.split()))