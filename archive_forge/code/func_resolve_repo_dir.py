import os
import sys
from subprocess import PIPE, STDOUT, Popen
import pytest
import zmq
def resolve_repo_dir(path):
    """Resolve a dir in the repo

    Resolved relative to zmq dir

    fallback on CWD (e.g. test run from repo, zmq installed, not -e)
    """
    resolved_path = os.path.join(os.path.dirname(zmq_dir), path)
    if not os.path.exists(resolved_path):
        resolved_path = path
    return resolved_path