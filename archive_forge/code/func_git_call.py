import sys
from dulwich.object_store import MissingObjectFinder, peel_sha
from dulwich.protocol import Protocol
from dulwich.server import (Backend, BackendRepo, ReceivePackHandler,
from .. import errors, trace
from ..controldir import ControlDir
from .mapping import decode_git_path, default_mapping
from .object_store import BazaarObjectStore, get_object_store
from .refs import get_refs_container
def git_call(environ, start_response):
    req = HTTPGitRequest(environ, start_response, dumb=False, handlers=DEFAULT_HANDLERS)
    return handler(req, backend, mat)