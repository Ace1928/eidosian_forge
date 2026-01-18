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
@gitlab.exceptions.on_http_error(gitlab.exceptions.GitlabMarkdownError)
def markdown(self, text: str, gfm: bool=False, project: Optional[str]=None, **kwargs: Any) -> str:
    """Render an arbitrary Markdown document.

        Args:
            text: The markdown text to render
            gfm: Render text using GitLab Flavored Markdown. Default is False
            project: Full path of a project used a context when `gfm` is True
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabMarkdownError: If the server cannot perform the request

        Returns:
            The HTML rendering of the markdown text.
        """
    post_data = {'text': text, 'gfm': gfm}
    if project is not None:
        post_data['project'] = project
    data = self.http_post('/markdown', post_data=post_data, **kwargs)
    if TYPE_CHECKING:
        assert not isinstance(data, requests.Response)
        assert isinstance(data['html'], str)
    return data['html']