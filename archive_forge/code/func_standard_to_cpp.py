from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
def standard_to_cpp(level):
    """Converts an integer level from the standard value to the cpp value.

  Args:
    level: int, a Python standard logging level.

  Raises:
    TypeError: Raised when level is not an integer.

  Returns:
    The corresponding integer level for use in cpp logging.
  """
    return absl_to_cpp(standard_to_absl(level))