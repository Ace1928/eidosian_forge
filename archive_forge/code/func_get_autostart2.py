from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def get_autostart2(self, entryid):
    if not self.module.check_mode:
        return self.find_entry(entryid).autostart()
    else:
        try:
            return self.find_entry(entryid).autostart()
        except Exception:
            return self.module.exit_json(changed=True)