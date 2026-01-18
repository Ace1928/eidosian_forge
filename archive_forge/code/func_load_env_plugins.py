import contextlib
import copy
import difflib
import importlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import (
import numpy as np
from gym.wrappers import (
from gym.wrappers.compatibility import EnvCompatibility
from gym.wrappers.env_checker import PassiveEnvChecker
from gym import Env, error, logger
def load_env_plugins(entry_point: str='gym.envs') -> None:
    for plugin in metadata.entry_points(group=entry_point):
        module, attr = (None, None)
        try:
            module, attr = (plugin.module, plugin.attr)
        except AttributeError:
            if ':' in plugin.value:
                module, attr = plugin.value.split(':', maxsplit=1)
            else:
                module, attr = (plugin.value, None)
        except Exception as e:
            warnings.warn(f'While trying to load plugin `{plugin}` from {entry_point}, an exception occurred: {e}')
            module, attr = (None, None)
        finally:
            if attr is None:
                raise error.Error(f'Gym environment plugin `{module}` must specify a function to execute, not a root module')
        context = namespace(plugin.name)
        if plugin.name.startswith('__') and plugin.name.endswith('__'):
            if plugin.name == '__root__' or plugin.name == '__internal__':
                context = contextlib.nullcontext()
            else:
                logger.warn(f'The environment namespace magic key `{plugin.name}` is unsupported. To register an environment at the root namespace you should specify the `__root__` namespace.')
        with context:
            fn = plugin.load()
            try:
                fn()
            except Exception as e:
                logger.warn(str(e))