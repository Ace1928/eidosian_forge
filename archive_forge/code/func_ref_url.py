import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union
from urllib.parse import urlparse
from wandb.errors.term import termwarn
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.hashutil import (
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
def ref_url(self) -> str:
    """Get a URL to this artifact entry.

        These URLs can be referenced by another artifact.

        Returns:
            (str): A URL representing this artifact entry.

        Examples:
            Basic usage
            ```
            ref_url = source_artifact.get_entry('file.txt').ref_url()
            derived_artifact.add_reference(ref_url)
            ```
        """
    if self._parent_artifact is None:
        raise NotImplementedError
    assert self._parent_artifact.id is not None
    return 'wandb-artifact://' + b64_to_hex_id(B64MD5(self._parent_artifact.id)) + '/' + self.path