from typing import Callable, Optional
import numpy as np
import modin.pandas as pd
from modin.config import NPartitions
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
from modin.core.storage_formats.pandas import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import get_current_execution
def compute_batch(self, postprocessor: Optional[Callable]=None, pass_partition_id: Optional[bool]=False, pass_output_id: Optional[bool]=False):
    """
        Run the completed pipeline + any postprocessing steps end to end.

        Parameters
        ----------
        postprocessor : Callable, default: None
            A postprocessing function to be applied to each output partition.
            The order of arguments passed is `df` (the partition), `output_id`
            (if `pass_output_id=True`), and `partition_id` (if `pass_partition_id=True`).
        pass_partition_id : bool, default: False
            Whether or not to pass the numerical partition id to the postprocessing function.
        pass_output_id : bool, default: False
            Whether or not to pass the output ID associated with output queries to the
            postprocessing function.

        Returns
        -------
        list or dict or DataFrame
            If output ids are specified, a dictionary mapping output id to the resulting dataframe
            is returned, otherwise, a list of the resulting dataframes is returned.
        """
    if len(self.outputs) == 0:
        ErrorMessage.single_warning('No outputs to compute. Returning an empty list. Please specify outputs by calling `add_query` with `is_output=True`.')
        return []
    if not self.is_output_id_specified and pass_output_id:
        raise ValueError('`pass_output_id` is set to True, but output ids have not been specified. ' + 'To pass output ids, please specify them using the `output_id` kwarg with pipeline.add_query')
    if self.is_output_id_specified:
        outs = {}
    else:
        outs = []
    modin_frame = self.df._query_compiler._modin_frame
    partitions = modin_frame._partition_mgr_cls.row_partitions(modin_frame._partitions)
    for node in self.outputs:
        partitions = self._complete_nodes(node.operators + [node], partitions)
        for part in partitions:
            part.drain_call_queue(num_splits=1)
        if postprocessor:
            output_partitions = []
            for partition_id, partition in enumerate(partitions):
                args = []
                if pass_output_id:
                    args.append(node.output_id)
                if pass_partition_id:
                    args.append(partition_id)
                output_partitions.append(partition.add_to_apply_calls(postprocessor, *args))
        else:
            output_partitions = [part.add_to_apply_calls(lambda df: df) for part in partitions]
        [part.drain_call_queue(num_splits=self.num_partitions) for part in output_partitions]
        if not self.is_output_id_specified:
            outs.append(output_partitions)
        else:
            outs[node.output_id] = output_partitions
    if self.is_output_id_specified:
        final_results = {}
        id_df_iter = outs.items()
    else:
        final_results = [None] * len(outs)
        id_df_iter = enumerate(outs)
    for id, df in id_df_iter:
        partitions = []
        for row_partition in df:
            partitions.append(row_partition.list_of_block_partitions)
        partitions = np.array(partitions)
        partition_mgr_class = PandasOnRayDataframe._partition_mgr_cls
        index, internal_rows = partition_mgr_class.get_indices(0, partitions)
        columns, internal_cols = partition_mgr_class.get_indices(1, partitions)
        result_modin_frame = PandasOnRayDataframe(partitions, index, columns, row_lengths=list(map(len, internal_rows)), column_widths=list(map(len, internal_cols)))
        query_compiler = PandasQueryCompiler(result_modin_frame)
        result_df = pd.DataFrame(query_compiler=query_compiler)
        final_results[id] = result_df
    return final_results