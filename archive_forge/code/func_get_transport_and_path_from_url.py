import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
def get_transport_and_path_from_url(url: str, config: Optional[Config]=None, operation: Optional[str]=None, **kwargs) -> Tuple[GitClient, str]:
    """Obtain a git client from a URL.

    Args:
      url: URL to open (a unicode string)
      config: Optional config object
      operation: Kind of operation that'll be performed; "pull" or "push"
      thin_packs: Whether or not thin packs should be retrieved
      report_activity: Optional callback for reporting transport
        activity.

    Returns:
      Tuple with client instance and relative path.

    """
    if config is not None:
        url = apply_instead_of(config, url, push=operation == 'push')
    return _get_transport_and_path_from_url(url, config=config, operation=operation, **kwargs)