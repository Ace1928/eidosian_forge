from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def log_critical(self, msg):
    """Log a CRITICAL message

        Convenience function to log a message to the default Logger

        Parameters
        ----------
        msg : str
            Message to be logged
        """
    self._log(self.logger.critical, msg)