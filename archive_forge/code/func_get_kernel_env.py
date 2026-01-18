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
def get_kernel_env(self, path: Optional[str], name: Optional[ModelName]=None) -> Dict[str, str]:
    """Return the environment variables that need to be set in the kernel

        Parameters
        ----------
        path : str
            the url path for the given session.
        name: ModelName(str), optional
            Here the name is likely to be the name of the associated file
            with the current kernel at startup time.
        """
    if name is not None:
        cwd = self.kernel_manager.cwd_for_path(path)
        path = os.path.join(cwd, name)
    assert isinstance(path, str)
    return {**os.environ, 'JPY_SESSION_NAME': path}