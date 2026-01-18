import copy
import os
import pickle
import warnings
import numpy as np
def prettyInfo(self):
    s = ''
    titles = []
    maxl = 0
    for i in range(len(self._info) - 1):
        ax = self._info[i]
        axs = ''
        if 'name' in ax:
            axs += '"%s"' % str(ax['name'])
        else:
            axs += '%d' % i
        if 'units' in ax:
            axs += ' (%s)' % str(ax['units'])
        titles.append(axs)
        if len(axs) > maxl:
            maxl = len(axs)
    for i in range(min(self.ndim, len(self._info) - 1)):
        ax = self._info[i]
        axs = titles[i]
        axs += '%s[%d] :' % (' ' * (maxl - len(axs) + 5 - len(str(self.shape[i]))), self.shape[i])
        if 'values' in ax:
            if self.shape[i] > 0:
                v0 = ax['values'][0]
                axs += '  values: [%g' % v0
                if self.shape[i] > 1:
                    v1 = ax['values'][-1]
                    axs += ' ... %g] (step %g)' % (v1, (v1 - v0) / (self.shape[i] - 1))
                else:
                    axs += ']'
            else:
                axs += '  values: []'
        if 'cols' in ax:
            axs += ' columns: '
            colstrs = []
            for c in range(len(ax['cols'])):
                col = ax['cols'][c]
                cs = str(col.get('name', c))
                if 'units' in col:
                    cs += ' (%s)' % col['units']
                colstrs.append(cs)
            axs += '[' + ', '.join(colstrs) + ']'
        s += axs + '\n'
    s += str(self._info[-1])
    return s