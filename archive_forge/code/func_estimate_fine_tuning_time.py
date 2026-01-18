from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def estimate_fine_tuning_time(df: pd.DataFrame) -> None:
    """
    Estimate the time it'll take to fine-tune the dataset
    """
    ft_format = infer_task_type(df)
    expected_time = 1.0
    if ft_format == 'classification':
        num_examples = len(df)
        expected_time = num_examples * 1.44
    else:
        size = df.memory_usage(index=True).sum()
        expected_time = size * 0.0515

    def format_time(time: float) -> str:
        if time < 60:
            return f'{round(time, 2)} seconds'
        elif time < 3600:
            return f'{round(time / 60, 2)} minutes'
        elif time < 86400:
            return f'{round(time / 3600, 2)} hours'
        else:
            return f'{round(time / 86400, 2)} days'
    time_string = format_time(expected_time + 140)
    sys.stdout.write(f"Once your model starts training, it'll approximately take {time_string} to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.\n")