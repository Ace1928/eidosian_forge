import functools
import re
import warnings
def next_minor(self):
    if self.prerelease and self.patch == 0:
        return Version(major=self.major, minor=self.minor, patch=0, partial=self.partial)
    else:
        return Version(major=self.major, minor=self.minor + 1, patch=0, partial=self.partial)