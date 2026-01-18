from typing import Any, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager
@cli.register_custom_action('SidekiqManager')
@exc.on_http_error(exc.GitlabGetError)
def job_stats(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Return statistics about the jobs performed.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the information couldn't be retrieved

        Returns:
            Statistics about the Sidekiq jobs performed
        """
    return self.gitlab.http_get('/sidekiq/job_stats', **kwargs)