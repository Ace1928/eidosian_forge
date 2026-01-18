from __future__ import annotations
import inspect
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Type
from typing import Union
from . import base
from . import url as _url
from .interfaces import DBAPIConnection
from .mock import create_mock_engine
from .. import event
from .. import exc
from .. import util
from ..pool import _AdhocProxiedConnection
from ..pool import ConnectionPoolEntry
from ..sql import compiler
from ..util import immutabledict
def engine_from_config(configuration: Dict[str, Any], prefix: str='sqlalchemy.', **kwargs: Any) -> Engine:
    """Create a new Engine instance using a configuration dictionary.

    The dictionary is typically produced from a config file.

    The keys of interest to ``engine_from_config()`` should be prefixed, e.g.
    ``sqlalchemy.url``, ``sqlalchemy.echo``, etc.  The 'prefix' argument
    indicates the prefix to be searched for.  Each matching key (after the
    prefix is stripped) is treated as though it were the corresponding keyword
    argument to a :func:`_sa.create_engine` call.

    The only required key is (assuming the default prefix) ``sqlalchemy.url``,
    which provides the :ref:`database URL <database_urls>`.

    A select set of keyword arguments will be "coerced" to their
    expected type based on string values.    The set of arguments
    is extensible per-dialect using the ``engine_config_types`` accessor.

    :param configuration: A dictionary (typically produced from a config file,
        but this is not a requirement).  Items whose keys start with the value
        of 'prefix' will have that prefix stripped, and will then be passed to
        :func:`_sa.create_engine`.

    :param prefix: Prefix to match and then strip from keys
        in 'configuration'.

    :param kwargs: Each keyword argument to ``engine_from_config()`` itself
        overrides the corresponding item taken from the 'configuration'
        dictionary.  Keyword arguments should *not* be prefixed.

    """
    options = {key[len(prefix):]: configuration[key] for key in configuration if key.startswith(prefix)}
    options['_coerce_config'] = True
    options.update(kwargs)
    url = options.pop('url')
    return create_engine(url, **options)