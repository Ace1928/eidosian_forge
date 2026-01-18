from __future__ import annotations
import contextlib
import datetime
from functools import partial
from functools import wraps
import json
import logging
from numbers import Number
import threading
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from decorator import decorate
from . import exception
from .api import BackendArguments
from .api import BackendFormatted
from .api import CachedValue
from .api import CacheMutex
from .api import CacheReturnType
from .api import CantDeserializeException
from .api import KeyType
from .api import MetaDataType
from .api import NO_VALUE
from .api import SerializedReturnType
from .api import Serializer
from .api import ValuePayload
from .backends import _backend_loader
from .backends import register_backend  # noqa
from .proxy import ProxyBackend
from .util import function_key_generator
from .util import function_multi_key_generator
from .util import repr_obj
from .. import Lock
from .. import NeedRegenerationException
from ..util import coerce_string_conf
from ..util import memoized_property
from ..util import NameRegistry
from ..util import PluginLoader
from ..util.typing import Self
def configure_from_config(self, config_dict, prefix):
    """Configure from a configuration dictionary
        and a prefix.

        Example::

            local_region = make_region()
            memcached_region = make_region()

            # regions are ready to use for function
            # decorators, but not yet for actual caching

            # later, when config is available
            myconfig = {
                "cache.local.backend":"dogpile.cache.dbm",
                "cache.local.arguments.filename":"/path/to/dbmfile.dbm",
                "cache.memcached.backend":"dogpile.cache.pylibmc",
                "cache.memcached.arguments.url":"127.0.0.1, 10.0.0.1",
            }
            local_region.configure_from_config(myconfig, "cache.local.")
            memcached_region.configure_from_config(myconfig,
                                                "cache.memcached.")

        """
    config_dict = coerce_string_conf(config_dict)
    return self.configure(config_dict['%sbackend' % prefix], expiration_time=config_dict.get('%sexpiration_time' % prefix, None), _config_argument_dict=config_dict, _config_prefix='%sarguments.' % prefix, wrap=config_dict.get('%swrap' % prefix, None), replace_existing_backend=config_dict.get('%sreplace_existing_backend' % prefix, False))