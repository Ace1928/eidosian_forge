from __future__ import absolute_import, division, print_function
def create_or_update_logprofile(self):
    """
        Creates or Update log profile.

        :return: deserialized log profile state dictionary
        """
    self.log('Creating log profile instance {0}'.format(self.name))
    try:
        params = LogProfileResource(location=self.location, locations=self.locations, categories=self.categories, retention_policy=RetentionPolicy(days=self.retention_policy['days'], enabled=self.retention_policy['enabled']) if self.retention_policy else None, storage_account_id=self.storage_account if self.storage_account else None, service_bus_rule_id=self.service_bus_rule_id if self.service_bus_rule_id else None, tags=self.tags)
        response = self.monitor_log_profiles_client.log_profiles.create_or_update(log_profile_name=self.name, parameters=params)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except HttpResponseError as exc:
        self.log('Error attempting to create/update log profile.')
        self.fail('Error creating/updating log profile: {0}'.format(str(exc)))
    return logprofile_to_dict(response)