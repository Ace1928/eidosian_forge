from __future__ import annotations
from fastapi import Depends, Request
from fastapi.background import BackgroundTasks
from lazyops.libs.fastapi_utils.utils import create_function_wrapper
from ..types import errors
from ..types.current_user import CurrentUser, UserRole
from ..types.security import Authorization, APIKey
from ..utils.lazy import logger
from typing import Optional, List, Annotated, Type, Union
def has_api_key(*args, api_key: APIKey, current_user: OptionalUser, **kwargs):
    """
        Checks if the api key is valid
        """
    if api_key not in api_keys:
        if allow_authorized_user and current_user and current_user.is_valid:
            return
        if verbose:
            logger.info(f'`{api_key}` is not a valid api key')
        if dry_run:
            return
        raise errors.InvalidAPIKeyException(detail=f'`{api_key}` is not a valid api key')