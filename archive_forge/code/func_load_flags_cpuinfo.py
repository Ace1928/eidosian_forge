import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
def load_flags_cpuinfo(self, magic_key):
    self.features_flags = self.get_cpuinfo_item(magic_key)