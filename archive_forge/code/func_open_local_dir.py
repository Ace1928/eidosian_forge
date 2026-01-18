import os
from ..controldir import ControlDir
from ..errors import NoRepositoryPresent, NotBranchError
from ..plugins.fastimport import exporter as fastexporter
from ..repository import InterRepository
from ..transport import get_transport_from_path
from . import LocalGitProber
from .dir import BareLocalGitControlDirFormat, LocalGitControlDirFormat
from .object_store import get_object_store
from .refs import get_refs_container, ref_to_branch_name
from .repository import GitRepository
def open_local_dir():
    try:
        git_path = os.environ['GIT_DIR']
    except KeyError:
        git_transport = get_transport_from_path('.')
        git_format = LocalGitProber().probe_transport(git_transport)
    else:
        if git_path.endswith('/.git'):
            git_format = LocalGitControlDirFormat()
            git_path = git_path[:-4]
        else:
            git_format = BareLocalGitControlDirFormat()
        git_transport = get_transport_from_path(git_path)
    return git_format.open(git_transport)