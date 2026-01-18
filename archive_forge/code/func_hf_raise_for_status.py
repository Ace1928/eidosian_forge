import re
from typing import Optional
from requests import HTTPError, Response
from ._fixes import JSONDecodeError
def hf_raise_for_status(response: Response, endpoint_name: Optional[str]=None) -> None:
    """
    Internal version of `response.raise_for_status()` that will refine a
    potential HTTPError. Raised exception will be an instance of `HfHubHTTPError`.

    This helper is meant to be the unique method to raise_for_status when making a call
    to the Hugging Face Hub.

    Example:
    ```py
        import requests
        from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

        response = get_session().post(...)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            print(str(e)) # formatted message
            e.request_id, e.server_message # details returned by server

            # Complete the error message with additional information once it's raised
            e.append_to_message("
`create_commit` expects the repository to exist.")
            raise
    ```

    Args:
        response (`Response`):
            Response from the server.
        endpoint_name (`str`, *optional*):
            Name of the endpoint that has been called. If provided, the error message
            will be more complete.

    <Tip warning={true}>

    Raises when the request has failed:

        - [`~utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it
            doesn't exist, because `repo_type` is not set correctly, or because the repo
            is `private` and you do not have access.
        - [`~utils.GatedRepoError`]
            If the repository exists but is gated and the user is not on the authorized
            list.
        - [`~utils.RevisionNotFoundError`]
            If the repository exists but the revision couldn't be find.
        - [`~utils.EntryNotFoundError`]
            If the repository exists but the entry (e.g. the requested file) couldn't be
            find.
        - [`~utils.BadRequestError`]
            If request failed with a HTTP 400 BadRequest error.
        - [`~utils.HfHubHTTPError`]
            If request failed for a reason not listed above.

    </Tip>
    """
    try:
        response.raise_for_status()
    except HTTPError as e:
        error_code = response.headers.get('X-Error-Code')
        error_message = response.headers.get('X-Error-Message')
        if error_code == 'RevisionNotFound':
            message = f'{response.status_code} Client Error.' + '\n\n' + f'Revision Not Found for url: {response.url}.'
            raise RevisionNotFoundError(message, response) from e
        elif error_code == 'EntryNotFound':
            message = f'{response.status_code} Client Error.' + '\n\n' + f'Entry Not Found for url: {response.url}.'
            raise EntryNotFoundError(message, response) from e
        elif error_code == 'GatedRepo':
            message = f'{response.status_code} Client Error.' + '\n\n' + f'Cannot access gated repo for url {response.url}.'
            raise GatedRepoError(message, response) from e
        elif error_message == 'Access to this resource is disabled.':
            message = f'{response.status_code} Client Error.' + '\n\n' + f'Cannot access repository for url {response.url}.' + '\n' + 'Access to this resource is disabled.'
            raise DisabledRepoError(message, response) from e
        elif error_code == 'RepoNotFound' or (response.status_code == 401 and response.request is not None and (response.request.url is not None) and (REPO_API_REGEX.search(response.request.url) is not None)):
            message = f'{response.status_code} Client Error.' + '\n\n' + f'Repository Not Found for url: {response.url}.' + '\nPlease make sure you specified the correct `repo_id` and `repo_type`.\nIf you are trying to access a private or gated repo, make sure you are authenticated.'
            raise RepositoryNotFoundError(message, response) from e
        elif response.status_code == 400:
            message = f'\n\nBad request for {endpoint_name} endpoint:' if endpoint_name is not None else '\n\nBad request:'
            raise BadRequestError(message, response=response) from e
        raise HfHubHTTPError(str(e), response=response) from e