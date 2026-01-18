from __future__ import absolute_import, division, print_function
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def get_unneeded_pkgs(base):
    query = libdnf5.rpm.PackageQuery(base)
    query.filter_installed()
    query.filter_unneeded()
    for pkg in query:
        yield pkg