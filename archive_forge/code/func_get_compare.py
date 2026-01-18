import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def get_compare(self, other, weights=None):
    """return an instance of CompareMeans with self and other

        Parameters
        ----------
        other : array_like or instance of DescrStatsW
            If array_like then this creates an instance of DescrStatsW with
            the given weights.
        weights : None or array
            weights are only used if other is not an instance of DescrStatsW

        Returns
        -------
        cm : instance of CompareMeans
            the instance has self attached as d1 and other as d2.

        See Also
        --------
        CompareMeans

        """
    if not isinstance(other, self.__class__):
        d2 = DescrStatsW(other, weights)
    else:
        d2 = other
    return CompareMeans(self, d2)