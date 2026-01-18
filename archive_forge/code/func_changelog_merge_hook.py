from ... import version_info  # noqa: F401
from ...hooks import install_lazy_named_hook
def changelog_merge_hook(merger):
    """Merger.merge_file_content hook for GNU-format ChangeLog files."""
    from ...plugins.changelog_merge.changelog_merge import ChangeLogMerger
    return ChangeLogMerger(merger)