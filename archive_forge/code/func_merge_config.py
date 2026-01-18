import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
@classmethod
def merge_config(cls, options: Dict[str, Any], gitlab_id: Optional[str]=None, config_files: Optional[List[str]]=None) -> 'Gitlab':
    """Create a Gitlab connection by merging configuration with
        the following precedence:

        1. Explicitly provided CLI arguments,
        2. Environment variables,
        3. Configuration files:
            a. explicitly defined config files:
                i. via the `--config-file` CLI argument,
                ii. via the `PYTHON_GITLAB_CFG` environment variable,
            b. user-specific config file,
            c. system-level config file,
        4. Environment variables always present in CI (CI_SERVER_URL, CI_JOB_TOKEN).

        Args:
            options: A dictionary of explicitly provided key-value options.
            gitlab_id: ID of the configuration section.
            config_files: List of paths to configuration files.
        Returns:
            (gitlab.Gitlab): A Gitlab connection.

        Raises:
            gitlab.config.GitlabDataError: If the configuration is not correct.
        """
    config = gitlab.config.GitlabConfigParser(gitlab_id=gitlab_id, config_files=config_files)
    url = options.get('server_url') or config.url or os.getenv('CI_SERVER_URL') or gitlab.const.DEFAULT_URL
    private_token, oauth_token, job_token = cls._merge_auth(options, config)
    return cls(url=url, private_token=private_token, oauth_token=oauth_token, job_token=job_token, ssl_verify=options.get('ssl_verify') or config.ssl_verify, timeout=options.get('timeout') or config.timeout, api_version=options.get('api_version') or config.api_version, per_page=options.get('per_page') or config.per_page, pagination=options.get('pagination') or config.pagination, order_by=options.get('order_by') or config.order_by, user_agent=options.get('user_agent') or config.user_agent)