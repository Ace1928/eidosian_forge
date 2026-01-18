import inspect
import re
import warnings
from functools import wraps
from itertools import chain
from typing import Any, Dict
from ._typing import CallableT
def smoothly_deprecate_use_auth_token(fn_name: str, has_token: bool, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Smoothly deprecate `use_auth_token` in the `huggingface_hub` codebase.

    The long-term goal is to remove any mention of `use_auth_token` in the codebase in
    favor of a unique and less verbose `token` argument. This will be done a few steps:

    0. Step 0: methods that require a read-access to the Hub use the `use_auth_token`
       argument (`str`, `bool` or `None`). Methods requiring write-access have a `token`
       argument (`str`, `None`). This implicit rule exists to be able to not send the
       token when not necessary (`use_auth_token=False`) even if logged in.

    1. Step 1: we want to harmonize everything and use `token` everywhere (supporting
       `token=False` for read-only methods). In order not to break existing code, if
       `use_auth_token` is passed to a function, the `use_auth_token` value is passed
       as `token` instead, without any warning.
       a. Corner case: if both `use_auth_token` and `token` values are passed, a warning
          is thrown and the `use_auth_token` value is ignored.

    2. Step 2: Once it is release, we should push downstream libraries to switch from
       `use_auth_token` to `token` as much as possible, but without throwing a warning
       (e.g. manually create issues on the corresponding repos).

    3. Step 3: After a transitional period (6 months e.g. until April 2023?), we update
       `huggingface_hub` to throw a warning on `use_auth_token`. Hopefully, very few
       users will be impacted as it would have already been fixed.
       In addition, unit tests in `huggingface_hub` must be adapted to expect warnings
       to be thrown (but still use `use_auth_token` as before).

    4. Step 4: After a normal deprecation cycle (3 releases ?), remove this validator.
       `use_auth_token` will definitely not be supported.
       In addition, we update unit tests in `huggingface_hub` to use `token` everywhere.

    This has been discussed in:
    - https://github.com/huggingface/huggingface_hub/issues/1094.
    - https://github.com/huggingface/huggingface_hub/pull/928
    - (related) https://github.com/huggingface/huggingface_hub/pull/1064
    """
    new_kwargs = kwargs.copy()
    use_auth_token = new_kwargs.pop('use_auth_token', None)
    if use_auth_token is not None:
        if has_token:
            warnings.warn(f'Both `token` and `use_auth_token` are passed to `{fn_name}` with non-None values. `token` is now the preferred argument to pass a User Access Token. `use_auth_token` value will be ignored.')
        else:
            new_kwargs['token'] = use_auth_token
    return new_kwargs