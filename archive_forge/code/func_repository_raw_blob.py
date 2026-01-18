from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project', ('sha',))
@exc.on_http_error(exc.GitlabGetError)
def repository_raw_blob(self, sha: str, streamed: bool=False, action: Optional[Callable[..., Any]]=None, chunk_size: int=1024, *, iterator: bool=False, **kwargs: Any) -> Optional[Union[bytes, Iterator[Any]]]:
    """Return the raw file contents for a blob.

        Args:
            sha: ID of the blob
            streamed: If True the data will be processed by chunks of
                `chunk_size` and each chunk is passed to `action` for
                treatment
            iterator: If True directly return the underlying response
                iterator
            action: Callable responsible of dealing with chunk of
                data
            chunk_size: Size of each chunk
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The blob content if streamed is False, None otherwise
        """
    path = f'/projects/{self.encoded_id}/repository/blobs/{sha}/raw'
    result = self.manager.gitlab.http_get(path, streamed=streamed, raw=True, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(result, requests.Response)
    return utils.response_content(result, streamed, action, chunk_size, iterator=iterator)