import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def allocation_candidate_list(self, resources=None, required=None, forbidden=None, limit=None, aggregate_uuids=None, member_of=None, may_print_to_stderr=False):
    cmd = 'allocation candidate list '
    cmd += self._allocation_candidates_option(resources, required, forbidden, aggregate_uuids, member_of)
    if limit is not None:
        cmd += ' --limit %d' % limit
    return self.openstack(cmd, use_json=True, may_print_to_stderr=may_print_to_stderr)