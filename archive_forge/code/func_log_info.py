from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def log_info(self, msg):
    """Log an INFO message

        Convenience function to log a message to the default Logger

        Parameters
        ----------
        msg : str
            Message to be logged
        """
    self._log(self.logger.info, msg)