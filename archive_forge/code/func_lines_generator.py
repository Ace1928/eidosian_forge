import pandas as pd
from .utils import series_to_line
def lines_generator(self):
    for num, (index, line) in enumerate(self._data.iterrows()):
        if self._group_feature_num is None:
            yield (num, series_to_line(line, self._sep) + '\n')
        else:
            yield (line.iloc[self._group_feature_num], series_to_line(line, self._sep) + '\n')