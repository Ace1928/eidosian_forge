import functools
import re
import warnings
def next_major(self):
    if self.prerelease and self.minor == self.patch == 0:
        return Version(major=self.major, minor=0, patch=0, partial=self.partial)
    else:
        return Version(major=self.major + 1, minor=0, patch=0, partial=self.partial)