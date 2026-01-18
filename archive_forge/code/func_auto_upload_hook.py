from ... import version_info  # noqa: F401
from ... import commands, config, hooks
def auto_upload_hook(params):
    import sys
    from ... import osutils, trace, transport, urlutils
    from .cmds import BzrUploader
    source_branch = params.branch
    conf = source_branch.get_config_stack()
    destination = conf.get('upload_location')
    if destination is None:
        return
    auto_upload = conf.get('upload_auto')
    if not auto_upload:
        return
    quiet = conf.get('upload_auto_quiet')
    if not quiet:
        display_url = urlutils.unescape_for_display(destination, osutils.get_terminal_encoding())
        trace.note('Automatically uploading to %s', display_url)
    to_transport = transport.get_transport(destination)
    last_revision = source_branch.last_revision()
    last_tree = source_branch.repository.revision_tree(last_revision)
    uploader = BzrUploader(source_branch, to_transport, sys.stdout, last_tree, last_revision, quiet=quiet)
    uploader.upload_tree()