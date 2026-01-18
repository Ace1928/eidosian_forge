import csv
import datetime
import os
def supported(self, date=None, result='codename'):
    """Get list of all supported Ubuntu distributions based on the given
        date."""
    if date is None:
        date = self._date
    distros = [self._format(result, x) for x in self._avail(date) if date <= x.eol or (x.eol_server is not None and date <= x.eol_server)]
    return distros