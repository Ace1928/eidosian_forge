import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
def migrate_ancient_formats(repo_transport):
    repo_transport = remove_readonly_transport_decorator(repo_transport)
    has_sqlite = repo_transport.has('git.db')
    has_tdb = repo_transport.has('git.tdb')
    if not has_sqlite or has_tdb:
        return
    try:
        repo_transport.mkdir('git')
    except FileExists:
        return
    if has_sqlite:
        SqliteGitCacheFormat().initialize(repo_transport.clone('git'))
        repo_transport.rename('git.db', 'git/idmap.db')
    elif has_tdb:
        TdbGitCacheFormat().initialize(repo_transport.clone('git'))
        repo_transport.rename('git.tdb', 'git/idmap.tdb')