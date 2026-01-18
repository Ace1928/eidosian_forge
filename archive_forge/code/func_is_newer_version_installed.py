from __future__ import absolute_import, division, print_function
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def is_newer_version_installed(base, spec):
    try:
        spec_nevra = next(iter(libdnf5.rpm.Nevra.parse(spec)))
    except RuntimeError:
        return False
    spec_name = spec_nevra.get_name()
    v = spec_nevra.get_version()
    r = spec_nevra.get_release()
    if not v or not r:
        return False
    spec_evr = '{}:{}-{}'.format(spec_nevra.get_epoch() or '0', v, r)
    query = libdnf5.rpm.PackageQuery(base)
    query.filter_installed()
    query.filter_name([spec_name])
    query.filter_evr([spec_evr], libdnf5.common.QueryCmp_GT)
    return query.size() > 0