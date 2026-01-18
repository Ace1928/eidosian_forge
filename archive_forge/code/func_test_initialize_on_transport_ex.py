from breezy import errors, tests
from breezy.tests.per_repository_reference import \
def test_initialize_on_transport_ex(self):
    base = self.make_branch('base')
    trans = self.get_transport('stacked')
    repo = self.initialize_and_check_on_transport(base, trans)
    self.assertEqual(base.repository._format.network_name(), repo._format.network_name())