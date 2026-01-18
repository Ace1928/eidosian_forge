from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def apply_optional_remediation(df: pd.DataFrame, remediation: Remediation, auto_accept: bool) -> tuple[pd.DataFrame, bool]:
    """
    This function will apply an optional remediation to a dataframe, based on the user input.
    """
    optional_applied = False
    input_text = f'- [Recommended] {remediation.optional_msg} [Y/n]: '
    if remediation.optional_msg is not None:
        if accept_suggestion(input_text, auto_accept):
            assert remediation.optional_fn is not None
            df = remediation.optional_fn(df)
            optional_applied = True
    if remediation.necessary_msg is not None:
        sys.stdout.write(f'- [Necessary] {remediation.necessary_msg}\n')
    return (df, optional_applied)