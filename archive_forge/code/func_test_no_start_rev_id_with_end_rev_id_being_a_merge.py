import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_no_start_rev_id_with_end_rev_id_being_a_merge(self):
    revs = log._generate_all_revisions(self.branch, None, '2.1.3', 'reverse', delayed_graph_generation=True)