from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def send_key_to_vm(self, vm_obj):
    key_event = None
    num_keys_returned = 0
    key_queue = []
    if self.params['keys_send']:
        for specified_key in self.params['keys_send']:
            key_found = False
            for keys in self.keys_hid_code:
                if isinstance(keys[0], tuple) and specified_key in keys[0] or (not isinstance(keys[0], tuple) and specified_key == keys[0]):
                    hid_code, modifiers = self.get_hid_from_key(specified_key)
                    key_event = self.get_key_event(hid_code, modifiers)
                    key_queue.append(key_event)
                    self.num_keys_send += 1
                    key_found = True
                    break
            if not key_found:
                self.module.fail_json(msg="keys_send parameter: '%s' in %s not supported." % (specified_key, self.params['keys_send']))
    if self.params['string_send']:
        for char in self.params['string_send']:
            key_found = False
            for keys in self.keys_hid_code:
                if isinstance(keys[0], tuple) and char in keys[0] or char == ' ':
                    hid_code, modifiers = self.get_hid_from_key(char)
                    key_event = self.get_key_event(hid_code, modifiers)
                    key_queue.append(key_event)
                    self.num_keys_send += 1
                    key_found = True
                    break
            if not key_found:
                self.module.fail_json(msg="string_send parameter: '%s' contains char: '%s' not supported." % (self.params['string_send'], char))
    if key_queue:
        try:
            num_keys_returned = self.send_key_events(vm_obj=vm_obj, key_queue=key_queue, sleep_time=self.module.params.get('sleep_time'))
            self.change_detected = True
        except vmodl.RuntimeFault as e:
            self.module.fail_json(msg='Failed to send key %s to virtual machine due to %s' % (key_event, to_native(e.msg)))
    sendkey_facts = self.get_sendkey_facts(vm_obj, num_keys_returned)
    if num_keys_returned != self.num_keys_send:
        results = {'changed': self.change_detected, 'failed': True, 'sendkey_info': sendkey_facts}
    else:
        results = {'changed': self.change_detected, 'failed': False, 'sendkey_info': sendkey_facts}
    return results