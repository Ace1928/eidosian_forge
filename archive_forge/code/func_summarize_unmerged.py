from io import StringIO
from ... import branch as _mod_branch
from ... import controldir, errors
from ... import forge as _mod_forge
from ... import log as _mod_log
from ... import missing as _mod_missing
from ... import msgeditor, urlutils
from ...commands import Command
from ...i18n import gettext
from ...option import ListOption, Option, RegistryOption
from ...trace import note, warning
def summarize_unmerged(local_branch, remote_branch, target, prerequisite_branch=None):
    """Generate a text description of the unmerged revisions in branch.

    :param branch: The proposed branch
    :param target: Target branch
    :param prerequisite_branch: Optional prerequisite branch
    :return: A string
    """
    log_format = _mod_log.log_formatter_registry.get_default(local_branch)
    to_file = StringIO()
    lf = log_format(to_file=to_file, show_ids=False, show_timezone='original')
    if prerequisite_branch:
        local_extra = _mod_missing.find_unmerged(remote_branch, prerequisite_branch, restrict='local')[0]
    else:
        local_extra = _mod_missing.find_unmerged(remote_branch, target, restrict='local')[0]
    if remote_branch.supports_tags():
        rev_tag_dict = remote_branch.tags.get_reverse_tag_dict()
    else:
        rev_tag_dict = {}
    for revision in _mod_missing.iter_log_revisions(local_extra, local_branch.repository, False, rev_tag_dict):
        lf.log_revision(revision)
    return to_file.getvalue()