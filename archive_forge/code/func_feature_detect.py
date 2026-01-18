import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_detect(self, names):
    """
        Return a list of CPU features that required to be detected
        sorted from the lowest to highest interest.
        """
    names = self.feature_get_til(names, 'implies_detect')
    detect = []
    for n in names:
        d = self.feature_supported[n]
        detect += d.get('detect', d.get('group', [n]))
    return detect