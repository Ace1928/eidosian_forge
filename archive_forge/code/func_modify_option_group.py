import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def modify_option_group(self, option_group_name, options_to_include=None, options_to_remove=None, apply_immediately=None):
    """
        Modifies an existing option group.

        :type option_group_name: string
        :param option_group_name: The name of the option group to be modified.
        Permanent options, such as the TDE option for Oracle Advanced Security
            TDE, cannot be removed from an option group, and that option group
            cannot be removed from a DB instance once it is associated with a
            DB instance

        :type options_to_include: list
        :param options_to_include: Options in this list are added to the option
            group or, if already present, the specified configuration is used
            to update the existing configuration.

        :type options_to_remove: list
        :param options_to_remove: Options in this list are removed from the
            option group.

        :type apply_immediately: boolean
        :param apply_immediately: Indicates whether the changes should be
            applied immediately, or during the next maintenance window for each
            instance associated with the option group.

        """
    params = {'OptionGroupName': option_group_name}
    if options_to_include is not None:
        self.build_complex_list_params(params, options_to_include, 'OptionsToInclude.member', ('OptionName', 'Port', 'DBSecurityGroupMemberships', 'VpcSecurityGroupMemberships', 'OptionSettings'))
    if options_to_remove is not None:
        self.build_list_params(params, options_to_remove, 'OptionsToRemove.member')
    if apply_immediately is not None:
        params['ApplyImmediately'] = str(apply_immediately).lower()
    return self._make_request(action='ModifyOptionGroup', verb='POST', path='/', params=params)