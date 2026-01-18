from breezy import errors, tests
from breezy.tests.per_repository_reference import \
def initialize_and_check_on_transport(self, base, trans):
    network_name = base.repository._format.network_name()
    result = self.bzrdir_format.initialize_on_transport_ex(trans, use_existing_dir=False, create_prefix=False, stacked_on='../base', stack_on_pwd=base.base, repo_format_name=network_name)
    result_repo, a_controldir, require_stacking, repo_policy = result
    self.addCleanup(result_repo.unlock)
    self.assertEqual(1, len(result_repo._fallback_repositories))
    return result_repo