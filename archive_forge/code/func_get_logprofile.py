from __future__ import absolute_import, division, print_function
def get_logprofile(self):
    """
        Gets the properties of the specified log profile.

        :return: log profile state dictionary
        """
    self.log('Checking if the log profile {0} is present'.format(self.name))
    response = None
    try:
        response = self.monitor_log_profiles_client.log_profiles.get(log_profile_name=self.name)
        self.log('Response : {0}'.format(response))
        self.log('log profile : {0} found'.format(response.name))
        return logprofile_to_dict(response)
    except HttpResponseError:
        self.log("Didn't find log profile {0}".format(self.name))
    return False