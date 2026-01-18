import sys
from dulwich.object_store import MissingObjectFinder, peel_sha
from dulwich.protocol import Protocol
from dulwich.server import (Backend, BackendRepo, ReceivePackHandler,
from .. import errors, trace
from ..controldir import ControlDir
from .mapping import decode_git_path, default_mapping
from .object_store import BazaarObjectStore, get_object_store
from .refs import get_refs_container
def serve_git_upload_pack(transport, host=None, port=None, inet=False):
    if not inet:
        raise errors.CommandError('git-receive-pack only works in inetd mode')
    backend = BzrBackend(transport)
    sys.exit(serve_command(UploadPackHandler, backend=backend))