from functools import wraps
def relative_update(self, inc=1):
    """
        Delta increment the internal counter

        Triggers ``call()``

        Parameters
        ----------
        inc: int
        """
    self.value += inc
    self.call()