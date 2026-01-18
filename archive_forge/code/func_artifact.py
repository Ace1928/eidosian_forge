from typing import Any, Callable, cast, Dict, Iterator, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RefreshMixin, RetrieveMixin
from gitlab.types import ArrayAttribute
@cli.register_custom_action('ProjectJob')
@exc.on_http_error(exc.GitlabGetError)
def artifact(self, path: str, streamed: bool=False, action: Optional[Callable[..., Any]]=None, chunk_size: int=1024, *, iterator: bool=False, **kwargs: Any) -> Optional[Union[bytes, Iterator[Any]]]:
    """Get a single artifact file from within the job's artifacts archive.

        Args:
            path: Path of the artifact
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
            GitlabGetError: If the artifacts could not be retrieved

        Returns:
            The artifacts if `streamed` is False, None otherwise.
        """
    path = f'{self.manager.path}/{self.encoded_id}/artifacts/{path}'
    result = self.manager.gitlab.http_get(path, streamed=streamed, raw=True, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(result, requests.Response)
    return utils.response_content(result, streamed, action, chunk_size, iterator=iterator)