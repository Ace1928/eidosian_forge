import csv
import datetime
import os
def supported_esm(self, date=None, result='codename'):
    """Get list of all ESM supported Ubuntu distributions based on the
        given date."""
    if date is None:
        date = self._date
    distros = [self._format(result, x) for x in self._avail(date) if x.eol_esm is not None and date <= x.eol_esm]
    return distros