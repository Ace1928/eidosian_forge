import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def image_id(image_name: str) -> Optional[str]:
    """Retreve the image id from the local docker daemon or remote registry."""
    if '@sha256:' in image_name:
        return image_name
    else:
        digests = shell(['inspect', image_name, '--format', '{{json .RepoDigests}}'])
        try:
            if digests is None:
                raise ValueError
            im_id: str = json.loads(digests)[0]
            return im_id
        except (ValueError, IndexError):
            return image_id_from_registry(image_name)