from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from modin.logging import ClassLogger
def force_materialization(self, get_ip: bool=False) -> 'BaseDataframeAxisPartition':
    """
        Materialize axis partitions into a single partition.

        Parameters
        ----------
        get_ip : bool, default: False
            Whether to get node ip address to a single partition or not.

        Returns
        -------
        BaseDataframeAxisPartition
            An axis partition containing only a single materialized partition.
        """
    materialized = self.apply(lambda x: x, num_splits=1, maintain_partitioning=False)
    return type(self)(materialized, get_ip=get_ip)