from __future__ import absolute_import, division, print_function
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
def get_srs(session):
    recs = session.xenapi.SR.get_all_records()
    if not recs:
        return None
    srs = change_keys(recs, key='name_label')
    return srs