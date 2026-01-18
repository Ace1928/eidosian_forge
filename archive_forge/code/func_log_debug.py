from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def log_debug(self, msg):
    """Log a DEBUG message

        Convenience function to log a message to the default Logger

        Parameters
        ----------
        msg : str
            Message to be logged
        """
    self._log(self.logger.debug, msg)