from __future__ import absolute_import, division, print_function
def delete_logprofile(self):
    """
        Deletes specified log profile.

        :return: True
        """
    self.log('Deleting the log profile instance {0}'.format(self.name))
    try:
        response = self.monitor_log_profiles_client.log_profiles.delete(log_profile_name=self.name)
    except HttpResponseError as e:
        self.log('Error attempting to delete the log profile.')
        self.fail('Error deleting the log profile: {0}'.format(str(e)))
    return True