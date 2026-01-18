import abc
import asyncio
import datetime
import functools
import importlib
import json
import logging
import os
import pkgutil
from abc import ABCMeta, abstractmethod
from base64 import b64decode
from collections import namedtuple
from collections.abc import MutableMapping, Mapping, Sequence
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._raylet import GcsClient
from ray._private.utils import split_address
import aiosignal  # noqa: F401
import ray._private.protobuf_compat
from frozenlist import FrozenList  # noqa: F401
from ray._private.utils import binary_to_hex, check_dashboard_dependencies_installed
def get_all_modules(module_type):
    """
    Get all importable modules that are subclass of a given module type.
    """
    logger.info(f'Get all modules by type: {module_type.__name__}')
    import ray.dashboard.modules
    should_only_load_minimal_modules = not check_dashboard_dependencies_installed()
    for module_loader, name, ispkg in pkgutil.walk_packages(ray.dashboard.modules.__path__, ray.dashboard.modules.__name__ + '.'):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            logger.info(f"Module {name} cannot be loaded because we cannot import all dependencies. Install this module using `pip install 'ray[default]'` for the full dashboard functionality. Error: {e}")
            if not should_only_load_minimal_modules:
                logger.info("Although `pip install 'ray[default]'` is downloaded, module couldn't be imported`")
                raise e
    imported_modules = []
    for m in module_type.__subclasses__():
        if not getattr(m, '__ray_dashboard_module_enable__', True):
            continue
        if should_only_load_minimal_modules and (not m.is_minimal_module()):
            continue
        imported_modules.append(m)
    logger.info(f'Available modules: {imported_modules}')
    return imported_modules