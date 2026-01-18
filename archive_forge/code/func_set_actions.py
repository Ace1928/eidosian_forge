from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def set_actions(self):
    """define what needs to be done"""
    actions = []
    modify = {}
    current = self.get_volume()
    self.volume_style = self.get_volume_style(current)
    if self.volume_style == 'flexgroup' and self.parameters.get('aggregate_name') is not None:
        self.module.fail_json(msg='Error: aggregate_name option cannot be used with FlexGroups.')
    cd_action = self.na_helper.get_cd_action(current, self.parameters)
    if cd_action == 'delete' or self.parameters['state'] == 'absent':
        return (['delete'] if cd_action == 'delete' else [], current, modify)
    if cd_action == 'create':
        if self.use_rest:
            rest_vserver.get_vserver_uuid(self.rest_api, self.parameters['vserver'], self.module, True)
        actions = ['create']
        if self.parameters.get('from_name'):
            current = self.get_volume(self.parameters['from_name'])
            rename = self.na_helper.is_rename_action(current, None)
            if rename is None:
                self.module.fail_json(msg='Error renaming volume: cannot find %s' % self.parameters['from_name'])
            if rename:
                cd_action = None
                actions = ['rename']
        elif self.parameters.get('from_vserver'):
            if self.use_rest:
                self.module.fail_json(msg='Error: ONTAP REST API does not support Rehosting Volumes')
            actions = ['rehost']
            self.na_helper.changed = True
    if self.parameters.get('snapshot_restore'):
        if 'create' in actions:
            self.module.fail_json(msg='Error restoring volume: cannot find parent: %s' % self.parameters['name'])
        actions.append('snapshot_restore')
        self.na_helper.changed = True
    self.validate_snaplock_changes(current)
    if cd_action is None and 'rehost' not in actions:
        modify = self.set_modify_dict(current)
        if modify:
            if not self.use_rest and modify.get('encrypt') is False and (not self.parameters.get('aggregate_name')):
                self.parameters['aggregate_name'] = current['aggregate_name']
            if self.use_rest and modify.get('encrypt') is False and (not modify.get('aggregate_name')):
                self.module.fail_json(msg='Error: unencrypting volume is only supported when moving the volume to another aggregate in REST.')
            actions.append('modify')
    if self.parameters.get('nas_application_template') is not None:
        application = self.get_application()
        changed = self.na_helper.changed
        app_component = self.create_nas_application_component() if self.parameters['state'] == 'present' else None
        modify_app = self.na_helper.get_modified_attributes(application, app_component)
        if modify_app:
            self.na_helper.changed = changed
            self.module.warn('Modifying an app is not supported at present: ignoring: %s' % str(modify_app))
    return (actions, current, modify)