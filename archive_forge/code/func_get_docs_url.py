from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
def get_docs_url(url: str) -> str:
    """
    Return the given docs.ansible.com URL updated to match the running ansible-test version, if it is not a pre-release version.
    The URL should be in the form: https://docs.ansible.com/ansible/devel/path/to/doc.html
    Where 'devel' will be replaced with the current version, unless it is a pre-release version.
    When run under a pre-release version, the URL will remain unchanged.
    This serves to provide a fallback URL for pre-release versions.
    It also makes searching the source for docs links easier, since a full URL is provided to this function.
    """
    url_prefix = 'https://docs.ansible.com/ansible-core/devel/'
    if not url.startswith(url_prefix):
        raise ValueError(f'URL "{url}" does not start with: {url_prefix}')
    ansible_version = get_ansible_version()
    if re.search('^[0-9.]+$', ansible_version):
        url_version = '.'.join(ansible_version.split('.')[:2])
        new_prefix = f'https://docs.ansible.com/ansible-core/{url_version}/'
        url = url.replace(url_prefix, new_prefix)
    return url