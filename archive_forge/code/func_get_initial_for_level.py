from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
def get_initial_for_level(level):
    """Gets the initial that should start the log line for the given level.

  It returns:
  - 'I' when: level < STANDARD_WARNING.
  - 'W' when: STANDARD_WARNING <= level < STANDARD_ERROR.
  - 'E' when: STANDARD_ERROR <= level < STANDARD_CRITICAL.
  - 'F' when: level >= STANDARD_CRITICAL.

  Args:
    level: int, a Python standard logging level.

  Returns:
    The first initial as it would be logged by the C++ logging module.
  """
    if level < STANDARD_WARNING:
        return 'I'
    elif level < STANDARD_ERROR:
        return 'W'
    elif level < STANDARD_CRITICAL:
        return 'E'
    else:
        return 'F'