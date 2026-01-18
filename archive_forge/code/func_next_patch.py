import functools
import re
import warnings
def next_patch(self):
    if self.prerelease:
        return Version(major=self.major, minor=self.minor, patch=self.patch, partial=self.partial)
    else:
        return Version(major=self.major, minor=self.minor, patch=self.patch + 1, partial=self.partial)