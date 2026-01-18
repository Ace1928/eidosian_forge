from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_snapshotschedule_parameters():
    """This method provide parameters required for the ansible snapshot
    schedule module on Unity"""
    return dict(name=dict(type='str'), id=dict(type='str'), type=dict(type='str', choices=['every_n_hours', 'every_day', 'every_n_days', 'every_week', 'every_month']), interval=dict(type='int'), hours_of_day=dict(type='list', elements='int'), day_interval=dict(type='int'), days_of_week=dict(type='list', elements='str', choices=['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY']), day_of_month=dict(type='int'), hour=dict(type='int'), minute=dict(type='int'), desired_retention=dict(type='int'), retention_unit=dict(type='str', choices=['hours', 'days'], default='hours'), auto_delete=dict(type='bool'), state=dict(required=True, type='str', choices=['present', 'absent']))