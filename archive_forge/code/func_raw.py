from typing import Any, Callable, Iterator, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
@cli.register_custom_action('ProjectArtifactManager', ('ref_name', 'artifact_path', 'job'))
@exc.on_http_error(exc.GitlabGetError)
def raw(self, ref_name: str, artifact_path: str, job: str, streamed: bool=False, action: Optional[Callable[[bytes], None]]=None, chunk_size: int=1024, *, iterator: bool=False, **kwargs: Any) -> Optional[Union[bytes, Iterator[Any]]]:
    """Download a single artifact file from a specific tag or branch from
        within the job's artifacts archive.

        Args:
            ref_name: Branch or tag name in repository. HEAD or SHA references
                are not supported.
            artifact_path: Path to a file inside the artifacts archive.
            job: The name of the job.
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
            The artifact if `streamed` is False, None otherwise.
        """
    path = f'{self.path}/{ref_name}/raw/{artifact_path}'
    result = self.gitlab.http_get(path, streamed=streamed, raw=True, job=job, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(result, requests.Response)
    return utils.response_content(result, streamed, action, chunk_size, iterator=iterator)