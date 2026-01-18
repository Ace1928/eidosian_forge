import sys
import datetime
from collections import namedtuple
def r_increment(self):
    if self.r_count < sys.maxint:
        self.r_count += 1
    if self.r_entered is None:
        self.r_entered = datetime.datetime.now()
    self.r_update()