from typing import Any, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager
@cli.register_custom_action('SidekiqManager')
@exc.on_http_error(exc.GitlabGetError)
def compound_metrics(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Return all available metrics and statistics.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the information couldn't be retrieved

        Returns:
            All available Sidekiq metrics and statistics
        """
    return self.gitlab.http_get('/sidekiq/compound_metrics', **kwargs)