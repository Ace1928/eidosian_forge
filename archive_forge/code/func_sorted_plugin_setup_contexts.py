import logging
import os
import json
from abc import ABC
from typing import List, Dict, Optional, Any, Type
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.uri_cache import URICache
from ray._private.runtime_env.constants import (
from ray.util.annotations import DeveloperAPI
from ray._private.utils import import_attr
def sorted_plugin_setup_contexts(self) -> List[PluginSetupContext]:
    """Get the sorted plugin setup contexts, sorted by increasing priority.

        Returns:
            The sorted plugin setup contexts.
        """
    return sorted(self.plugins.values(), key=lambda x: x.priority)