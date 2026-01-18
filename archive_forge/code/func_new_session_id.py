import os
import pathlib
import uuid
from typing import Any, Dict, List, NewType, Optional, Union, cast
from dataclasses import dataclass, fields
from jupyter_core.utils import ensure_async
from tornado import web
from traitlets import Instance, TraitError, Unicode, validate
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.traittypes import InstanceFromClasses
def new_session_id(self) -> str:
    """Create a uuid for a new session"""
    return str(uuid.uuid4())