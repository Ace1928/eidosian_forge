from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def log_error(self, msg):
    """Log an ERROR message

        Convenience function to log a message to the default Logger

        Parameters
        ----------
        msg : str
            Message to be logged
        """
    self._log(self.logger.error, msg)