import breezy.ui
from breezy.tests.per_repository_reference import \
def test_check_file_graph_across_external_boundary_ok(self):
    tree = self.make_branch_and_tree('base')
    self.build_tree(['base/file'])
    tree.add(['file'], ids=[b'file-id'])
    rev1_id = tree.commit('one')
    referring = self.make_branch_and_tree('referring')
    readonly_base = self.readonly_repository(tree.branch.repository)
    referring.branch.repository.add_fallback_repository(readonly_base)
    local_tree = referring.branch.create_checkout('local')
    self.build_tree_contents([('local/file', b'change')])
    rev2_id = local_tree.commit('two')
    check_result = referring.branch.repository.check(referring.branch.repository.all_revision_ids())
    check_result.report_results(verbose=False)
    self.assertFalse('inconsistent parents' in self.get_log())