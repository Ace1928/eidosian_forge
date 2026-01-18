import csv
import datetime
import os
def elts_supported(self, date=None, result='codename'):
    """Get list of all Extended LTS supported Debian distributions based on
        the given date."""
    if date is None:
        date = self._date
    distros = [self._format(result, x) for x in self._avail(date) if (x.eol_lts is not None and date > x.eol_lts) and (x.eol_elts is not None and date <= x.eol_elts)]
    return distros