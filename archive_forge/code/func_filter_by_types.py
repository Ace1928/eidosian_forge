from abc import ABC, abstractmethod
from typing import Callable, Dict, Hashable, List, Optional, Union
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType
@abstractmethod
def filter_by_types(self, types: List[Hashable]) -> 'ModinDataframe':
    """
        Allow the user to specify a type or set of types by which to filter the columns.

        Parameters
        ----------
        types : list of hashables
            The types to filter columns by.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe with only the columns whose dtypes appear in `types`.
        """
    pass