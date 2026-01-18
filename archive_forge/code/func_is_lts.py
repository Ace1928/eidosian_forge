import csv
import datetime
import os
def is_lts(self, codename):
    """Is codename an LTS release?"""
    distros = [x for x in self._releases if x.series == codename]
    if not distros:
        return False
    return 'LTS' in distros[0].version