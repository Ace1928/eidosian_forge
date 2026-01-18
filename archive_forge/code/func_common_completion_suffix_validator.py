from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def common_completion_suffix_validator(df: pd.DataFrame) -> Remediation:
    """
    This validator will suggest to add a common suffix to the completion if one doesn't already exist in case of classification or conditional generation.
    """
    error_msg = None
    immediate_msg = None
    optional_msg = None
    optional_fn = None
    ft_type = infer_task_type(df)
    if ft_type == 'open-ended generation' or ft_type == 'classification':
        return Remediation(name='common_suffix')
    common_suffix = get_common_xfix(df.completion, xfix='suffix')
    if (df.completion == common_suffix).all():
        error_msg = f'All completions are identical: `{common_suffix}`\nEnsure completions are different, otherwise the model will just repeat `{common_suffix}`'
        return Remediation(name='common_suffix', error_msg=error_msg)
    suggested_suffix = ' [END]'
    suffix_options = ['\n', '.', ' END', '***', '+++', '&&&', '$$$', '@@@', '%%%']
    for suffix_option in suffix_options:
        if df.completion.str.contains(suffix_option, regex=False).any():
            continue
        suggested_suffix = suffix_option
        break
    display_suggested_suffix = suggested_suffix.replace('\n', '\\n')

    def add_suffix(x: Any, suffix: Any) -> Any:
        x['completion'] += suffix
        return x
    if common_suffix != '':
        common_suffix_new_line_handled = common_suffix.replace('\n', '\\n')
        immediate_msg = f'\n- All completions end with suffix `{common_suffix_new_line_handled}`'
        if len(common_suffix) > 10:
            immediate_msg += f'. This suffix seems very long. Consider replacing with a shorter suffix, such as `{display_suggested_suffix}`'
        if df.completion.str[:-len(common_suffix)].str.contains(common_suffix, regex=False).any():
            immediate_msg += f'\n  WARNING: Some of your completions contain the suffix `{common_suffix}` more than once. We suggest that you review your completions and add a unique ending'
    else:
        immediate_msg = '\n- Your data does not contain a common ending at the end of your completions. Having a common ending string appended to the end of the completion makes it clearer to the fine-tuned model where the completion should end. See https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset for more detail and examples.'
    if common_suffix == '':
        optional_msg = f'Add a suffix ending `{display_suggested_suffix}` to all completions'

        def optional_fn(x: Any) -> Any:
            return add_suffix(x, suggested_suffix)
    return Remediation(name='common_completion_suffix', immediate_msg=immediate_msg, optional_msg=optional_msg, optional_fn=optional_fn, error_msg=error_msg)