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
def cmd_capabilities(self, outf, argv):
    outf.write(b'\n'.join([c.encode() for c in CAPABILITIES]) + b'\n\n')