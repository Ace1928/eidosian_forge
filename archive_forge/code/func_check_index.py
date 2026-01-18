from . import utils
import numpy as np
import pandas as pd
import warnings
def check_index(data, copy=False):
    """Ensure that the data index is unique in a pandas DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    copy : bool, optional (default: True)
        If True, return a modified copy of the data. Otherwise modify it in place.

    Returns
    -------
    data : pd.DataFrame
        Sanitized data
    """
    if not hasattr(data, 'index'):
        warnings.warn('scprep.sanitize.check_index only accepts pandas input', UserWarning)
        return data
    duplicated = data.index.duplicated()
    if np.any(duplicated):
        new_index = list(data.index)
        is_mi = isinstance(data.index, pd.MultiIndex)
        for idx in np.unique(data.index[duplicated]):
            if is_mi:
                rename = np.all([data.index.get_level_values(data.index.names[i]) == idx[i] for i in range(len(idx))], axis=0)
            else:
                rename = data.index == idx
            rename = np.argwhere(rename).flatten()
            for i, idx in enumerate(rename):
                if i == 0:
                    continue
                if is_mi:
                    new_index[idx] = new_index[idx][:-1] + ('{}.{}'.format(new_index[idx][-1], i),)
                else:
                    new_index[idx] = '{}.{}'.format(new_index[idx], i)
            print_new_index = ', '.join([str(new_index[r]) for r in rename])
            warnings.warn('Renamed {} copies of index {} to ({})'.format(len(rename), new_index[rename[0]], print_new_index), RuntimeWarning)
        if copy:
            data = data.copy()
        if is_mi:
            new_index = pd.MultiIndex.from_tuples(new_index, names=data.index.names)
        data.index = new_index
    return data