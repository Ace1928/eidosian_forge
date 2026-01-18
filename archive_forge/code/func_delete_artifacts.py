from typing import Any, Callable, cast, Dict, Iterator, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RefreshMixin, RetrieveMixin
from gitlab.types import ArrayAttribute
@cli.register_custom_action('ProjectJob')
@exc.on_http_error(exc.GitlabCreateError)
def delete_artifacts(self, **kwargs: Any) -> None:
    """Delete artifacts of a job.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the request could not be performed
        """
    path = f'{self.manager.path}/{self.encoded_id}/artifacts'
    self.manager.gitlab.http_delete(path, **kwargs)