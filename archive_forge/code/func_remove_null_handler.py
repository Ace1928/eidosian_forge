import logging
import sys
def remove_null_handler():
    """Remove the null handler from the Dulwich loggers.

    If a caller wants to set up logging using something other than
    default_logging_config, calling this function first is a minor optimization
    to avoid the overhead of using the _NullHandler.
    """
    _DULWICH_LOGGER.removeHandler(_NULL_HANDLER)