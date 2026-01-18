from typing import Any, cast, Dict, List, Optional, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
from .custom_attributes import UserCustomAttributeManager  # noqa: F401
from .events import UserEventManager  # noqa: F401
from .personal_access_tokens import UserPersonalAccessTokenManager  # noqa: F401
@cli.register_custom_action('User')
@exc.on_http_error(exc.GitlabBanError)
def ban(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Ban the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabBanError: If the user could not be banned

        Returns:
            Whether the user has been banned
        """
    path = f'/users/{self.encoded_id}/ban'
    server_data = self.manager.gitlab.http_post(path, **kwargs)
    if server_data:
        self._attrs['state'] = 'banned'
    return server_data