from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def make_changelog_merger(self, base_text, this_text, other_text):
    builder = self.make_builder()
    clog = builder.add_file(builder.root(), 'ChangeLog', base_text, True, file_id=b'clog-id')
    builder.change_contents(clog, other=other_text, this=this_text)
    merger = builder.make_merger(merge.Merge3Merger, ['ChangeLog'])
    merger.this_branch.get_config().set_user_option('changelog_merge_files', 'ChangeLog')
    merge_hook_params = merge.MergeFileHookParams(merger, ['ChangeLog', 'ChangeLog', 'ChangeLog'], None, 'file', 'file', 'conflict')
    changelog_merger = changelog_merge.ChangeLogMerger(merger)
    return (changelog_merger, merge_hook_params)