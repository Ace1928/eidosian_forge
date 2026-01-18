import time
def interval_elapsed(self):
    """Check whether it is time to update plot.
        Returns
        -------
        Boolean value of whethe to update now
        """
    return time.time() - self.last_update > self.display_freq