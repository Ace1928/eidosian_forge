from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import tempfile
import typing as t
from .constants import (
from .locale_util import (
from .io import (
from .config import (
from .util import (
from .util_common import (
from .ansible_util import (
from .containers import (
from .data import (
from .payload import (
from .ci import (
from .host_configs import (
from .connections import (
from .provisioning import (
from .content_config import (
def download_results(args: EnvironmentConfig, con: Connection, content_root: str, success: bool) -> None:
    """Download results from a delegated controller."""
    remote_results_root = os.path.join(content_root, data_context().content.results_path)
    local_test_root = os.path.dirname(os.path.join(data_context().content.root, data_context().content.results_path))
    remote_test_root = os.path.dirname(remote_results_root)
    remote_results_name = os.path.basename(remote_results_root)
    make_dirs(local_test_root)
    with tempfile.NamedTemporaryFile(prefix='ansible-test-result-', suffix='.tgz') as result_file:
        try:
            con.create_archive(chdir=remote_test_root, name=remote_results_name, dst=result_file, exclude=ResultType.TMP.name)
        except SubprocessError as ex:
            if success:
                raise
            display.warning(f'Failed to download results while handling an exception: {ex}')
        else:
            result_file.seek(0)
            local_con = LocalConnection(args)
            local_con.extract_archive(chdir=local_test_root, src=result_file)