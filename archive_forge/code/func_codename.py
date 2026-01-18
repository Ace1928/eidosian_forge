import csv
import datetime
import os
def codename(self, release, date=None, default=None):
    """Map 'unstable', 'testing', etc. to their codenames."""
    if release == 'unstable':
        codename = self.devel(date)
    elif release == 'testing':
        codename = self.testing(date)
    elif release == 'stable':
        codename = self.stable(date)
    elif release == 'oldstable':
        codename = self.old(date)
    else:
        codename = default
    return codename