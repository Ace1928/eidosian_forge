import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def allocation_candidate_granular(self, groups, group_policy=None, limit=None):
    cmd = 'allocation candidate list '
    for suffix, req_group in groups.items():
        if suffix:
            cmd += ' --group %s' % suffix
        cmd += self._allocation_candidates_option(**req_group)
        if limit is not None:
            cmd += ' --limit %d' % limit
        if group_policy is not None:
            cmd += ' --group-policy %s' % group_policy
    return self.openstack(cmd, use_json=True)