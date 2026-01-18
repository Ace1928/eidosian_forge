from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def schedule_modify_required(self, schedule_details):
    """Check if the desired snapshot schedule state is different from
            existing snapshot schedule state
            :param schedule_details: The dict containing snapshot schedule
             details
            :return: Boolean value to indicate if modification is needed
        """
    if schedule_details['rules'][0]['is_auto_delete'] and self.module.params['desired_retention'] and (self.module.params['auto_delete'] is None):
        self.module.fail_json(msg='Desired retention cannot be specified when auto_delete is true')
    if schedule_details['rules'][0]['retention_time'] and self.module.params['auto_delete']:
        self.module.fail_json(msg='auto_delete cannot be specified when existing desired retention is set')
    desired_rule_type = get_schedule_value(self.module.params['type'])
    existing_rule_string = schedule_details['rules'][0]['type'].split('.')[1]
    existing_rule_type = utils.ScheduleTypeEnum[existing_rule_string]._get_properties()['value']
    modified = False
    if desired_rule_type != existing_rule_type:
        self.module.fail_json(msg='Modification of rule type is not allowed.')
    duration_in_sec = convert_retention_to_seconds(self.module.params['desired_retention'], self.module.params['retention_unit'])
    if not duration_in_sec:
        duration_in_sec = schedule_details['rules'][0]['retention_time']
    if duration_in_sec and duration_in_sec != schedule_details['rules'][0]['retention_time']:
        modified = True
    elif self.module.params['auto_delete'] is not None and self.module.params['auto_delete'] != schedule_details['rules'][0]['is_auto_delete']:
        modified = True
    if self.module.params['minute'] is not None and self.module.params['minute'] != schedule_details['rules'][0]['minute']:
        modified = True
    if not modified and desired_rule_type == 0:
        if self.module.params['interval'] and self.module.params['interval'] != schedule_details['rules'][0]['interval']:
            modified = True
    elif not modified and desired_rule_type == 1:
        if self.module.params['hours_of_day'] and set(self.module.params['hours_of_day']) != set(schedule_details['rules'][0]['hours']):
            modified = True
    elif not modified and desired_rule_type == 2:
        if self.module.params['day_interval'] and self.module.params['day_interval'] != schedule_details['rules'][0]['interval'] or (self.module.params['hour'] is not None and self.module.params['hour'] != schedule_details['rules'][0]['hours'][0]):
            modified = True
    elif not modified and desired_rule_type == 3:
        days = schedule_details['rules'][0]['days_of_week']['DayOfWeekEnumList']
        existing_days = list()
        for day in days:
            temp = day.split('.')
            existing_days.append(temp[1])
        if self.module.params['days_of_week'] and set(self.module.params['days_of_week']) != set(existing_days) or (self.module.params['hour'] is not None and self.module.params['hour'] != schedule_details['rules'][0]['hours'][0]):
            modified = True
    elif not modified and desired_rule_type == 4:
        if self.module.params['day_of_month'] and self.module.params['day_of_month'] != schedule_details['rules'][0]['days_of_month'][0] or (self.module.params['hour'] is not None and self.module.params['hour'] != schedule_details['rules'][0]['hours'][0]):
            modified = True
    LOG.info('Modify Flag: %s', modified)
    return modified