import re
import numpy as np
import pytest
from mpl_toolkits.axisartist.angle_helper import (
def dms2float(degrees, minutes=0, seconds=0):
    return degrees + minutes / 60.0 + seconds / 3600.0