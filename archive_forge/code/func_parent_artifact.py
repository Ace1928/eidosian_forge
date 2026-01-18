import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union
from urllib.parse import urlparse
from wandb.errors.term import termwarn
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.hashutil import (
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
def parent_artifact(self) -> 'Artifact':
    """Get the artifact to which this artifact entry belongs.

        Returns:
            (PublicArtifact): The parent artifact
        """
    if self._parent_artifact is None:
        raise NotImplementedError
    return self._parent_artifact