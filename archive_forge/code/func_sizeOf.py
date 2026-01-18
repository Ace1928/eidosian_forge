import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
def sizeOf(text):
    return self.fontMetrics.boundingRect(text).height() + padding