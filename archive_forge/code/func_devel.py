import csv
import datetime
import os
def devel(self, date=None, result='codename'):
    """Get latest development distribution based on the given date."""
    if date is None:
        date = self._date
    distros = [x for x in self._avail(date) if x.release is None or (date < x.release and (x.eol is None or date <= x.eol))]
    if len(distros) < 2:
        raise DistroDataOutdated()
    return self._format(result, distros[-2])