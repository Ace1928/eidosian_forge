from abc import ABC, abstractmethod
from typing import Callable, Dict, Hashable, List, Optional, Union
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType
@abstractmethod
def from_labels(self) -> 'ModinDataframe':
    """
        Move the row labels into the data at position 0, and sets the row labels to the positional notation.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the row labels moved into the data.

        Notes
        -----
        In the case that the dataframe has hierarchical labels, all label "levels" are inserted into the dataframe
        in the order they occur in the labels, with the outermost being in position 0.
        """
    pass