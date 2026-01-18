from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_backup_policy(self):
    """
        Creates or updates backup policy.

        :return: ProtectionPolicyResource
        """
    self.log('Creating backup policy {0} for vault {1} in resource group {2}'.format(self.name, self.vault_name, self.resource_group))
    self.log('Creating backup policy in progress')
    response = None
    try:
        instant_rp_details = None
        dt = datetime.utcnow()
        dt = datetime(dt.year, dt.month, dt.day, 0, 0)
        schedule_run_times_as_datetimes = []
        schedule_run_time = self.schedule_run_time
        try:
            if 0 <= schedule_run_time <= 23:
                schedule_run_times_as_datetimes = [dt.replace(hour=schedule_run_time)]
            else:
                raise ValueError('Paramater schedule_run_time {0} is badly formed must be on the 24 hour scale'.format(schedule_run_time))
            if self.schedule_run_frequency == 'Weekly' and self.instant_recovery_snapshot_retention != 5:
                raise ValueError('Paramater instant_recovery_snapshot_retention was {0} but must be 5 when schedule_run_frequency is Weekly'.format(self.instant_recovery_snapshot_retention))
            if self.schedule_run_frequency == 'Weekly' and (not 1 <= self.weekly_retention_count <= 5163):
                raise ValueError('Paramater weekly_retention_count was {0} but must be between 1 and 5163 when schedule_run_frequency is Weekly'.format(self.weekly_retention_count))
            if self.schedule_run_frequency == 'Daily' and (not 7 <= self.daily_retention_count <= 9999):
                raise ValueError('Paramater daily_retention_count was {0} but must be between 7 and 9999 when schedule_run_frequency is Daily'.format(self.daily_retention_count))
        except ValueError as e:
            self.results['changed'] = False
            self.fail(e)
        schedule_policy = self.recovery_services_backup_models.SimpleSchedulePolicy(schedule_run_frequency=self.schedule_run_frequency, schedule_run_days=self.schedule_days, schedule_run_times=schedule_run_times_as_datetimes, schedule_weekly_frequency=self.schedule_weekly_frequency)
        daily_retention_schedule = None
        weekly_retention_schedule = None
        if self.daily_retention_count and self.schedule_run_frequency == 'Daily':
            retention_duration = self.recovery_services_backup_models.RetentionDuration(count=self.daily_retention_count, duration_type='Days')
            daily_retention_schedule = self.recovery_services_backup_models.DailyRetentionSchedule(retention_times=schedule_run_times_as_datetimes, retention_duration=retention_duration)
        if self.weekly_retention_count:
            retention_duration = self.recovery_services_backup_models.RetentionDuration(count=self.weekly_retention_count, duration_type='Weeks')
            weekly_retention_schedule = self.recovery_services_backup_models.WeeklyRetentionSchedule(days_of_the_week=self.schedule_days, retention_times=schedule_run_times_as_datetimes, retention_duration=retention_duration)
        retention_policy = self.recovery_services_backup_models.LongTermRetentionPolicy(daily_schedule=daily_retention_schedule, weekly_schedule=weekly_retention_schedule)
        policy_definition = None
        if self.backup_management_type == 'AzureIaasVM':
            AzureIaaSVMProtectionPolicy = self.recovery_services_backup_models.AzureIaaSVMProtectionPolicy
            policy_definition = AzureIaaSVMProtectionPolicy(instant_rp_details=instant_rp_details, schedule_policy=schedule_policy, retention_policy=retention_policy, instant_rp_retention_range_in_days=self.instant_recovery_snapshot_retention, time_zone=self.time_zone)
        if policy_definition:
            policy_resource = self.recovery_services_backup_models.ProtectionPolicyResource(properties=policy_definition)
            response = self.recovery_services_backup_client.protection_policies.create_or_update(vault_name=self.vault_name, resource_group_name=self.resource_group, policy_name=self.name, parameters=policy_resource)
    except Exception as e:
        self.log('Error attempting to create the backup policy.')
        self.fail('Error creating the backup policy {0} for vault {1} in resource group {2}. Error Reads: {3}'.format(self.name, self.vault_name, self.resource_group, e))
    return response