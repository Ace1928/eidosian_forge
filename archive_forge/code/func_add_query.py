from typing import Callable, Optional
import numpy as np
import modin.pandas as pd
from modin.config import NPartitions
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
from modin.core.storage_formats.pandas import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import get_current_execution
def add_query(self, func: Callable, is_output: bool=False, repartition_after: bool=False, fan_out: bool=False, pass_partition_id: bool=False, reduce_fn: Optional[Callable]=None, output_id: Optional[int]=None):
    """
        Add a query to the current pipeline.

        Parameters
        ----------
        func : Callable
            DataFrame query to perform.
        is_output : bool, default: False
            Whether this query should be designated as an output query. If `True`, the output of
            this query is passed both to the next query and directly to postprocessing.
        repartition_after : bool, default: False
            Whether the dataframe should be repartitioned after this query. Currently,
            repartitioning is only supported if there is 1 partition prior.
        fan_out : bool, default: False
            Whether to fan out this node. If True and only 1 partition is passed as input, the
            partition is replicated `self.num_partitions` (default: `NPartitions.get`) times,
            and the function is called on each. The `reduce_fn` must also be specified.
        pass_partition_id : bool, default: False
            Whether to pass the numerical partition id to the query.
        reduce_fn : Callable, default: None
            The reduce function to apply if `fan_out` is set to True. This takes the
            `self.num_partitions` (default: `NPartitions.get`) partitions that result from this
            query, and combines them into 1 partition.
        output_id : int, default: None
            An id to assign to this node if it is an output.

        Notes
        -----
        Use `pandas` for any module level functions inside `func` since it operates directly on
        partitions.
        """
    if not is_output and output_id is not None:
        raise ValueError('Output ID cannot be specified for non-output node.')
    if is_output:
        if not self.is_output_id_specified and output_id is not None:
            if len(self.outputs) != 0:
                raise ValueError('Output ID must be specified for all nodes.')
        if output_id is None and self.is_output_id_specified:
            raise ValueError('Output ID must be specified for all nodes.')
    self.query_list.append(PandasQuery(func, is_output, repartition_after, fan_out, pass_partition_id, reduce_fn, output_id))
    if is_output:
        self.outputs.append(self.query_list[-1])
        if output_id is not None:
            self.is_output_id_specified = True
        self.outputs[-1].operators = self.query_list[:-1]
        self.query_list = []