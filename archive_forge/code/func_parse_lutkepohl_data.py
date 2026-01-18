from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import asbytes
from statsmodels.tools.validation import array_like, int_like
import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.tsa.tsatools as tsa
def parse_lutkepohl_data(path):
    """
    Parse data files from Lütkepohl (2005) book

    Source for data files: www.jmulti.de
    """
    from collections import deque
    from datetime import datetime
    import re
    regex = re.compile(asbytes('<(.*) (\\w)([\\d]+)>.*'))
    with open(path, 'rb') as f:
        lines = deque(f)
    to_skip = 0
    while asbytes('*/') not in lines.popleft():
        to_skip += 1
    while True:
        to_skip += 1
        line = lines.popleft()
        m = regex.match(line)
        if m:
            year, freq, start_point = m.groups()
            break
    data = pd.read_csv(path, delimiter='\\s+', header=to_skip + 1).to_records(index=False)
    n = len(data)
    start_point = int(start_point)
    year = int(year)
    offsets = {asbytes('Q'): frequencies.BQuarterEnd(), asbytes('M'): frequencies.BMonthEnd(), asbytes('A'): frequencies.BYearEnd()}
    offset = offsets[freq]
    inc = offset * (start_point - 1)
    start_date = offset.rollforward(datetime(year, 1, 1)) + inc
    offset = offsets[freq]
    date_range = pd.date_range(start=start_date, freq=offset, periods=n)
    return (data, date_range)