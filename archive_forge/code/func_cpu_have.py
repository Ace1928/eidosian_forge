import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
def cpu_have(self, feature_name):
    map_names = self.features_map.get(feature_name, feature_name)
    if isinstance(map_names, str):
        return map_names in self.features_flags
    for f in map_names:
        if f in self.features_flags:
            return True
    return False