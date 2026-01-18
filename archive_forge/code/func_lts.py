import csv
import datetime
import os
def lts(self, date=None, result='codename'):
    """Get latest long term support (LTS) Ubuntu distribution based on the
        given date."""
    if date is None:
        date = self._date
    distros = [x for x in self._releases if x.version.find('LTS') >= 0 and x.release <= date <= x.eol]
    if not distros:
        raise DistroDataOutdated()
    return self._format(result, distros[-1])