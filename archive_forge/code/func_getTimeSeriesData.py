import string
from datetime import datetime
import numpy as np
import pandas as pd
def getTimeSeriesData(nper=None, freq='B'):
    return {c: makeTimeSeries(nper, freq) for c in getCols(_K)}