import os
import warnings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import ray
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
def convert_pandas_to_torch_tensor(data_batch: pd.DataFrame, columns: Optional[Union[List[str], List[List[str]]]]=None, column_dtypes: Optional[Union[torch.dtype, List[torch.dtype]]]=None, unsqueeze: bool=True) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Converts a Pandas dataframe to a torch Tensor or list of torch Tensors.

    The format of the return type will match the format of ``columns``. If a
    list of columns is provided, the return type will be a single tensor. If
    ``columns`` is a list of lists, then the return type will be a list of
    tensors.

    Args:
        data_batch: The pandas dataframe to convert to a
            torch tensor.
        columns:
            The names of the columns in the dataframe to include in the
            torch tensor. If this arg is a List[List[str]], then the return
            type will be a List of tensors. This is useful for multi-input
            models. If None, then use all columns in the ``data_batch``.
        column_dtype: The
            torch dtype to use for the tensor. If set to None,
            then automatically infer the dtype.
        unsqueeze: If set to True, the tensors
            will be unsqueezed (reshaped to (N, 1)) before being concatenated into
            the final tensor. Otherwise, they will be left as is, that is
            (N, ). Defaults to True.

    Returns:
        Either a torch tensor of size (N, len(columns)) where N is the
        number of rows in the ``data_batch`` Dataframe, or a list of
        tensors, where the size of item i is (N, len(columns[i])).

    """
    multi_input = columns and isinstance(columns[0], (list, tuple))
    if not multi_input and column_dtypes and (type(column_dtypes) != torch.dtype):
        raise TypeError(f'If `columns` is a list of strings, `column_dtypes` must be None or a single `torch.dtype`.Got {type(column_dtypes)} instead.')
    columns = columns if columns else []

    def tensorize(vals, dtype):
        """This recursive function allows to convert pyarrow List dtypes
        to multi-dimensional tensors."""
        if isinstance(vals, pd.api.extensions.ExtensionArray):
            vals = vals.to_numpy()
        if vals.dtype.type is np.object_:
            tensors = [tensorize(x, dtype) for x in vals]
            try:
                return torch.stack(tensors)
            except RuntimeError:
                return torch.nested_tensor(tensors)
        else:
            return torch.as_tensor(vals, dtype=dtype)

    def get_tensor_for_columns(columns, dtype):
        feature_tensors = []
        if columns:
            batch = data_batch[columns]
        else:
            batch = data_batch
        for col in batch.columns:
            col_vals = batch[col].values
            try:
                t = tensorize(col_vals, dtype=dtype)
            except Exception:
                raise ValueError(f'Failed to convert column {col} to a Torch Tensor of dtype {dtype}. See above exception chain for the exact failure.')
            if unsqueeze:
                t = t.unsqueeze(1)
            feature_tensors.append(t)
        if len(feature_tensors) > 1:
            feature_tensor = torch.cat(feature_tensors, dim=1)
        else:
            feature_tensor = feature_tensors[0]
        return feature_tensor
    if multi_input:
        if type(column_dtypes) not in [list, tuple]:
            column_dtypes = [column_dtypes] * len(columns)
        return [get_tensor_for_columns(columns=subcolumns, dtype=dtype) for subcolumns, dtype in zip(columns, column_dtypes)]
    else:
        return get_tensor_for_columns(columns=columns, dtype=column_dtypes)