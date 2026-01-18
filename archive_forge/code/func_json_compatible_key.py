import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
def json_compatible_key(key: str) -> str:
    """As defined in :pep:`566#json-compatible-metadata`"""
    return key.lower().replace('-', '_')