import sys
import time
from io import StringIO
from . import branch as _mod_branch
from . import controldir, errors
from . import hooks as _mod_hooks
from . import osutils, urlutils
from .bzr import bzrdir
from .errors import (NoRepositoryPresent, NotBranchError, NotLocalUrl,
from .missing import find_unmerged
def show_component_info(control, repository, branch=None, working=None, verbose=1, outfile=None):
    """Write info about all bzrdir components to stdout"""
    if outfile is None:
        outfile = sys.stdout
    if verbose is False:
        verbose = 1
    if verbose is True:
        verbose = 2
    layout = describe_layout(repository, branch, working, control)
    format = describe_format(control, repository, branch, working)
    outfile.write('{} (format: {})\n'.format(layout, format))
    _show_location_info(gather_location_info(control=control, repository=repository, branch=branch, working=working), outfile)
    if branch is not None:
        _show_related_info(branch, outfile)
    if verbose == 0:
        return
    _show_format_info(control, repository, branch, working, outfile)
    _show_locking_info(repository, branch, working, outfile)
    _show_control_dir_info(control, outfile)
    if branch is not None:
        _show_missing_revisions_branch(branch, outfile)
    if working is not None:
        _show_missing_revisions_working(working, outfile)
        _show_working_stats(working, outfile)
    elif branch is not None:
        _show_missing_revisions_branch(branch, outfile)
    if branch is not None:
        show_committers = verbose >= 2
        stats = _show_branch_stats(branch, show_committers, outfile)
    elif repository is not None:
        stats = repository.gather_stats()
    if branch is None and working is None and (repository is not None):
        _show_repository_info(repository, outfile)
    if repository is not None:
        _show_repository_stats(repository, stats, outfile)