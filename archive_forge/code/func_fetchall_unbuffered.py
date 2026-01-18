import re
import warnings
from . import err
def fetchall_unbuffered(self):
    """
        Fetch all, implemented as a generator, which isn't to standard,
        however, it doesn't make sense to return everything in a list, as that
        would use ridiculous memory for large result sets.
        """
    return iter(self.fetchone, None)