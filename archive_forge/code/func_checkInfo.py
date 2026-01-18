import copy
import os
import pickle
import warnings
import numpy as np
def checkInfo(self):
    info = self._info
    if info is None:
        if self._data is None:
            return
        else:
            self._info = [{} for i in range(self.ndim + 1)]
            return
    else:
        try:
            info = list(info)
        except:
            raise Exception('Info must be a list of axis specifications')
        if len(info) < self.ndim + 1:
            info.extend([{}] * (self.ndim + 1 - len(info)))
        elif len(info) > self.ndim + 1:
            raise Exception('Info parameter must be list of length ndim+1 or less.')
        for i in range(len(info)):
            if not isinstance(info[i], dict):
                if info[i] is None:
                    info[i] = {}
                else:
                    raise Exception('Axis specification must be Dict or None')
            if i < self.ndim and 'values' in info[i]:
                if type(info[i]['values']) is list:
                    info[i]['values'] = np.array(info[i]['values'])
                elif type(info[i]['values']) is not np.ndarray:
                    raise Exception('Axis values must be specified as list or ndarray')
                if info[i]['values'].ndim != 1 or info[i]['values'].shape[0] != self.shape[i]:
                    raise Exception('Values array for axis %d has incorrect shape. (given %s, but should be %s)' % (i, str(info[i]['values'].shape), str((self.shape[i],))))
            if i < self.ndim and 'cols' in info[i]:
                if not isinstance(info[i]['cols'], list):
                    info[i]['cols'] = list(info[i]['cols'])
                if len(info[i]['cols']) != self.shape[i]:
                    raise Exception('Length of column list for axis %d does not match data. (given %d, but should be %d)' % (i, len(info[i]['cols']), self.shape[i]))
        self._info = info