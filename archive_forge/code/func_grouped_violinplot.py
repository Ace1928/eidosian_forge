import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def grouped_violinplot(data1, data2, normalize=False, group_names=['data1', 'data2'], filename=None):
    """
    Plot a grouped violinplot to compare two datasets

    The datasets can be normalized by the median and standard deviation of data1.

    Parameters
    ----------
    data1: DataFrame
        Data set, columns = variable names
    data2: DataFrame
        Data set, columns = variable names
    normalize : bool, optional
        Normalize both datasets by the median and standard deviation of data1
    group_names : list, optional
        Names used in the legend
    filename: string, optional
        Filename used to save the figure
    """
    assert isinstance(data1, pd.DataFrame)
    assert isinstance(data2, pd.DataFrame)
    assert isinstance(normalize, bool)
    assert isinstance(group_names, list)
    assert isinstance(filename, (type(None), str))
    data = _get_grouped_data(data1, data2, normalize, group_names)
    plt.figure()
    sns.violinplot(data=data, hue='set', y='value', x='columns', order=data1.columns, split=True)
    plt.gca().legend().set_title('')
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()