from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
@cli.register_custom_action('ProjectPipelineSchedule')
@exc.on_http_error(exc.GitlabOwnershipError)
def take_ownership(self, **kwargs: Any) -> None:
    """Update the owner of a pipeline schedule.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabOwnershipError: If the request failed
        """
    path = f'{self.manager.path}/{self.encoded_id}/take_ownership'
    server_data = self.manager.gitlab.http_post(path, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(server_data, dict)
    self._update_attrs(server_data)