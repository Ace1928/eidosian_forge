import json
import functools
from lazyops.types.models import BaseModel, validator
from lazyops.types.classprops import lazyproperty
from lazyops.types.static import RESPONSE_SUCCESS_CODES
from lazyops.types.resources import BaseResource, ResourceType, ResponseResource, ResponseResourceType
from lazyops.types.errors import ClientError, fatal_exception
from lazyops.imports._aiohttpx import aiohttpx, resolve_aiohttpx
from lazyops.imports._backoff import backoff, require_backoff
from lazyops.configs.base import DefaultSettings
from lazyops.utils.logs import default_logger as logger
from lazyops.utils.serialization import ObjectEncoder
from typing import Optional, Dict, List, Any, Type, Callable
@classmethod
def init_settings(cls, settings_cls: Optional[SettingType]=None, settings: Optional[DefaultSettings]=None, **kwargs):
    """
        Initializes the settings
        """
    if settings_cls is not None:
        cls.settings_cls = settings_cls
        if cls.settings is None:
            cls.settings = cls.settings_cls()
    if settings is not None:
        cls.settings = settings
    if cls.settings is None and cls.settings_cls:
        cls.settings = cls.settings_cls()
    if cls.settings and kwargs:
        cls.settings.update_config(**kwargs)