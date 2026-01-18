from __future__ import annotations
import datetime
import os
import platform
import sys
import typing as t
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...docker_util import (
from ...constants import (
from ...ci import (
from ...timeout import (
def show_dump_env(args: EnvConfig) -> None:
    """Show information about the current environment and/or write the information to disk."""
    if not args.show and (not args.dump):
        return
    container_id = get_docker_container_id()
    data = dict(ansible=dict(version=get_ansible_version()), docker=get_docker_details(args), container_id=container_id, environ=os.environ.copy(), location=dict(pwd=os.environ.get('PWD', None), cwd=os.getcwd()), git=get_ci_provider().get_git_details(args), platform=dict(datetime=datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), platform=platform.platform(), uname=platform.uname()), python=dict(executable=sys.executable, version=platform.python_version()), interpreters=get_available_python_versions())
    if args.show:
        verbose = {'docker': 3, 'docker.executable': 0, 'environ': 2, 'platform.uname': 1}
        show_dict(data, verbose)
    if args.dump and (not args.explain):
        write_json_test_results(ResultType.BOT, 'data-environment.json', data)