import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def has_file(path_or_repo: Union[str, os.PathLike], filename: str, revision: Optional[str]=None, proxies: Optional[Dict[str, str]]=None, token: Optional[Union[bool, str]]=None, **deprecated_kwargs):
    """
    Checks if a repo contains a given file without downloading it. Works for remote repos and local folders.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
    this repo, but will return False for regular connection errors.

    </Tip>
    """
    use_auth_token = deprecated_kwargs.pop('use_auth_token', None)
    if use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
        if token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        token = use_auth_token
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))
    url = hf_hub_url(path_or_repo, filename=filename, revision=revision)
    headers = build_hf_headers(token=token, user_agent=http_user_agent())
    r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=10)
    try:
        hf_raise_for_status(r)
        return True
    except GatedRepoError as e:
        logger.error(e)
        raise EnvironmentError(f'{path_or_repo} is a gated repository. Make sure to request access at https://huggingface.co/{path_or_repo} and pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`.') from e
    except RepositoryNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'.")
    except RevisionNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions.")
    except requests.HTTPError:
        return False