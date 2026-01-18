from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def lun_actions(self, app_current, actions, results, scope, app_modify, app_modify_warning):
    lun_cd_action, lun_modify, lun_rename = (None, None, None)
    lun_path, from_lun_path = (None, None)
    from_name = self.parameters.get('from_name')
    if self.rest_app and app_current:
        lun_path = self.get_lun_path_from_backend(self.parameters['name'])
        if from_name is not None:
            from_lun_path = self.get_lun_path_from_backend(from_name)
    current = self.get_lun(self.parameters['name'], lun_path)
    self.set_uuid(current)
    if current is not None and lun_path is None:
        lun_path = current['path']
    lun_cd_action = self.na_helper.get_cd_action(current, self.parameters)
    if lun_cd_action == 'create' and from_name is not None:
        old_lun = self.get_lun(from_name, from_lun_path)
        lun_rename = self.na_helper.is_rename_action(old_lun, current)
        if lun_rename is None:
            self.module.fail_json(msg='Error renaming lun: %s does not exist' % from_name)
        if lun_rename:
            current = old_lun
            if from_lun_path is None:
                from_lun_path = current['path']
            head, _sep, tail = from_lun_path.rpartition(from_name)
            if tail:
                self.module.fail_json(msg='Error renaming lun: %s does not match lun_path %s' % (from_name, from_lun_path))
            self.set_uuid(current)
            lun_path = head + self.parameters['name']
            lun_cd_action = None
            actions.append('lun_rename')
            app_modify_warning = None
    if lun_cd_action is not None:
        actions.append('lun_%s' % lun_cd_action)
    if lun_cd_action is None and self.parameters['state'] == 'present':
        current.pop('name', None)
        lun_modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if lun_modify:
            actions.append('lun_modify')
            results['lun_modify'] = dict(lun_modify)
            app_modify_warning = None
    if lun_cd_action and self.rest_app and app_current:
        msg = 'This module does not support %s a LUN by name %s a SAN application.' % ('adding', 'to') if lun_cd_action == 'create' else ('removing', 'from')
        if scope == 'auto':
            self.module.warn(msg + ".  scope=%s, assuming 'application'" % scope)
            if not app_modify:
                self.na_helper.changed = False
        elif scope == 'lun':
            self.module.fail_json(msg=msg + '.  scope=%s.' % scope)
        lun_cd_action = None
    self.check_for_errors(lun_cd_action, current, lun_modify)
    return (lun_path, from_lun_path, lun_cd_action, lun_rename, lun_modify, app_modify_warning)