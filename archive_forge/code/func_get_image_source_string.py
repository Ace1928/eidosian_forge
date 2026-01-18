import enum
import logging
import os
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import wandb
import wandb.docker as docker
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch import utils
from wandb.sdk.lib.runid import generate_id
from .errors import LaunchError
from .utils import LOG_PREFIX, recursive_macro_sub
def get_image_source_string(self) -> str:
    """Returns a unique string identifying the source of an image."""
    if self.source == LaunchSource.LOCAL:
        assert isinstance(self.uri, str)
        return self.uri
    elif self.source == LaunchSource.JOB:
        assert self._job_artifact is not None
        return f'{self._job_artifact.name}:v{self._job_artifact.version}'
    elif self.source == LaunchSource.GIT:
        assert isinstance(self.uri, str)
        ret = self.uri
        if self.git_version:
            ret += self.git_version
        return ret
    elif self.source == LaunchSource.WANDB:
        assert isinstance(self.uri, str)
        return self.uri
    elif self.source == LaunchSource.DOCKER:
        assert isinstance(self.docker_image, str)
        _logger.debug('')
        return self.docker_image
    else:
        raise LaunchError('Unknown source type when determing image source string')