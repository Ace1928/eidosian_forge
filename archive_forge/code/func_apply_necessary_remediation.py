from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def apply_necessary_remediation(df: OptionalDataFrameT, remediation: Remediation) -> OptionalDataFrameT:
    """
    This function will apply a necessary remediation to a dataframe, or print an error message if one exists.
    """
    if remediation.error_msg is not None:
        sys.stderr.write(f'\n\nERROR in {remediation.name} validator: {remediation.error_msg}\n\nAborting...')
        sys.exit(1)
    if remediation.immediate_msg is not None:
        sys.stdout.write(remediation.immediate_msg)
    if remediation.necessary_fn is not None:
        df = remediation.necessary_fn(df)
    return df