import contextlib
import os
from dulwich.refs import SymrefLoop
from .. import branch as _mod_branch
from .. import errors as brz_errors
from .. import osutils, trace, urlutils
from ..controldir import (BranchReferenceLoop, ControlDir, ControlDirFormat,
from ..transport import (FileExists, NoSuchFile, do_catching_redirections,
from .mapping import decode_git_path, encode_git_path
from .push import GitPushResult
from .transportgit import OBJECTDIR, TransportObjectStore
def branch_names(self):
    from .refs import ref_to_branch_name
    ret = []
    for ref in self.get_refs_container().keys():
        try:
            branch_name = ref_to_branch_name(ref)
        except UnicodeDecodeError:
            trace.warning('Ignoring branch %r with unicode error ref', ref)
            continue
        except ValueError:
            continue
        ret.append(branch_name)
    return ret