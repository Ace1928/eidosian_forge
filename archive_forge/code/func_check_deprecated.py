from __future__ import annotations as _annotations
import warnings
from contextlib import contextmanager
from typing import (
from pydantic_core import core_schema
from typing_extensions import (
from ..aliases import AliasGenerator
from ..config import ConfigDict, ExtraValues, JsonDict, JsonEncoder, JsonSchemaExtraCallable
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20
def check_deprecated(config_dict: ConfigDict) -> None:
    """Check for deprecated config keys and warn the user.

    Args:
        config_dict: The input config.
    """
    deprecated_removed_keys = V2_REMOVED_KEYS & config_dict.keys()
    deprecated_renamed_keys = V2_RENAMED_KEYS.keys() & config_dict.keys()
    if deprecated_removed_keys or deprecated_renamed_keys:
        renamings = {k: V2_RENAMED_KEYS[k] for k in sorted(deprecated_renamed_keys)}
        renamed_bullets = [f'* {k!r} has been renamed to {v!r}' for k, v in renamings.items()]
        removed_bullets = [f'* {k!r} has been removed' for k in sorted(deprecated_removed_keys)]
        message = '\n'.join(['Valid config keys have changed in V2:'] + renamed_bullets + removed_bullets)
        warnings.warn(message, UserWarning)