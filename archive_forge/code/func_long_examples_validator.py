from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def long_examples_validator(df: pd.DataFrame) -> Remediation:
    """
    This validator will suggest to the user to remove examples that are too long.
    """
    immediate_msg = None
    optional_msg = None
    optional_fn = None
    ft_type = infer_task_type(df)
    if ft_type != 'open-ended generation':

        def get_long_indexes(d: pd.DataFrame) -> Any:
            long_examples = d.apply(lambda x: len(x.prompt) + len(x.completion) > 10000, axis=1)
            return d.reset_index().index[long_examples].tolist()
        long_indexes = get_long_indexes(df)
        if len(long_indexes) > 0:
            immediate_msg = f"\n- There are {len(long_indexes)} examples that are very long. These are rows: {long_indexes}\nFor conditional generation, and for classification the examples shouldn't be longer than 2048 tokens."
            optional_msg = f'Remove {len(long_indexes)} long examples'

            def optional_fn(x: Any) -> Any:
                long_indexes_to_drop = get_long_indexes(x)
                if long_indexes != long_indexes_to_drop:
                    sys.stdout.write(f'The indices of the long examples has changed as a result of a previously applied recommendation.\nThe {len(long_indexes_to_drop)} long examples to be dropped are now at the following indices: {long_indexes_to_drop}\n')
                return x.drop(long_indexes_to_drop)
    return Remediation(name='long_examples', immediate_msg=immediate_msg, optional_msg=optional_msg, optional_fn=optional_fn)