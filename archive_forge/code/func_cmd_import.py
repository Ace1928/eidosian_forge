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
def cmd_import(self, outf, argv):
    if 'fastimport' in CAPABILITIES:
        raise Exception("install fastimport for 'import' command support")
    ref = argv[1].encode('utf-8')
    dest_branch_name = ref_to_branch_name(ref)
    if dest_branch_name == 'master':
        dest_branch_name = None
    remote_branch = self.remote_dir.open_branch(name=dest_branch_name)
    exporter = fastexporter.BzrFastExporter(remote_branch, outf=outf, ref=ref, checkpoint=None, import_marks_file=None, export_marks_file=None, revision=None, verbose=None, plain_format=True, rewrite_tags=False)
    exporter.run()