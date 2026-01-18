from ... import ui
from ...i18n import gettext
from ...merge import Merger
from ...progress import ProgressPhase
from ...trace import note
from ..vf_repository import install_revision
def install_bundle(repository, bundle_reader):
    custom_install = getattr(bundle_reader, 'install', None)
    if custom_install is not None:
        return custom_install(repository)
    with repository.lock_write(), ui.ui_factory.nested_progress_bar() as pb:
        real_revisions = bundle_reader.real_revisions
        for i, revision in enumerate(reversed(real_revisions)):
            pb.update(gettext('Install revisions'), i, len(real_revisions))
            if repository.has_revision(revision.revision_id):
                continue
            cset_tree = bundle_reader.revision_tree(repository, revision.revision_id)
            install_revision(repository, revision, cset_tree)