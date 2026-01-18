from __future__ import annotations
import random
import asyncio
import logging
import contextlib
from enum import Enum
from lazyops.libs.kops.base import *
from lazyops.libs.kops.config import KOpsSettings
from lazyops.libs.kops.utils import cached, DillSerializer, SignalHandler
from lazyops.libs.kops._kopf import kopf
from lazyops.types import lazyproperty
from lazyops.utils import logger
from typing import List, Dict, Union, Any, Optional, Callable, TYPE_CHECKING
import lazyops.libs.kops.types as t
import lazyops.libs.kops.atypes as at
@property
def ingresses(cls) -> 'SyncClient.NetworkingV1Api':
    return cls.session.ingresses