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
def describe_layout(repository=None, branch=None, tree=None, control=None):
    """Convert a control directory layout into a user-understandable term

    Common outputs include "Standalone tree", "Repository branch" and
    "Checkout".  Uncommon outputs include "Unshared repository with trees"
    and "Empty control directory"
    """
    if branch is None and control is not None:
        try:
            branch_reference = control.get_branch_reference()
        except NotBranchError:
            pass
        else:
            if branch_reference is not None:
                return 'Dangling branch reference'
    if repository is None:
        return 'Empty control directory'
    if branch is None and tree is None:
        if repository.is_shared():
            phrase = 'Shared repository'
        else:
            phrase = 'Unshared repository'
        extra = []
        if repository.make_working_trees():
            extra.append('trees')
        if len(control.branch_names()) > 0:
            extra.append('colocated branches')
        if extra:
            phrase += ' with ' + ' and '.join(extra)
        return phrase
    else:
        if repository.is_shared():
            independence = 'Repository '
        else:
            independence = 'Standalone '
        if tree is not None:
            phrase = 'tree'
        else:
            phrase = 'branch'
        if branch is None and tree is not None:
            phrase = 'branchless tree'
        elif tree is not None and tree.controldir.control_url != branch.controldir.control_url:
            independence = ''
            phrase = 'Lightweight checkout'
        elif branch.get_bound_location() is not None:
            if independence == 'Standalone ':
                independence = ''
            if tree is None:
                phrase = 'Bound branch'
            else:
                phrase = 'Checkout'
        if independence != '':
            phrase = phrase.lower()
        return '{}{}'.format(independence, phrase)