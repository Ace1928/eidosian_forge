import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def reset_db_parameter_group(self, db_parameter_group_name, reset_all_parameters=None, parameters=None):
    """
        Modifies the parameters of a DB parameter group to the
        engine/system default value. To reset specific parameters
        submit a list of the following: `ParameterName` and
        `ApplyMethod`. To reset the entire DB parameter group, specify
        the `DBParameterGroup` name and `ResetAllParameters`
        parameters. When resetting the entire group, dynamic
        parameters are updated immediately and static parameters are
        set to `pending-reboot` to take effect on the next DB instance
        restart or `RebootDBInstance` request.

        :type db_parameter_group_name: string
        :param db_parameter_group_name:
        The name of the DB parameter group.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type reset_all_parameters: boolean
        :param reset_all_parameters: Specifies whether ( `True`) or not (
            `False`) to reset all parameters in the DB parameter group to
            default values.
        Default: `True`

        :type parameters: list
        :param parameters: An array of parameter names, values, and the apply
            method for the parameter update. At least one parameter name,
            value, and apply method must be supplied; subsequent arguments are
            optional. A maximum of 20 parameters may be modified in a single
            request.
        **MySQL**

        Valid Values (for Apply method): `immediate` | `pending-reboot`

        You can use the immediate value with dynamic parameters only. You can
            use the `pending-reboot` value for both dynamic and static
            parameters, and changes are applied when DB instance reboots.

        **Oracle**

        Valid Values (for Apply method): `pending-reboot`

        """
    params = {'DBParameterGroupName': db_parameter_group_name}
    if reset_all_parameters is not None:
        params['ResetAllParameters'] = str(reset_all_parameters).lower()
    if parameters is not None:
        self.build_complex_list_params(params, parameters, 'Parameters.member', ('ParameterName', 'ParameterValue', 'Description', 'Source', 'ApplyType', 'DataType', 'AllowedValues', 'IsModifiable', 'MinimumEngineVersion', 'ApplyMethod'))
    return self._make_request(action='ResetDBParameterGroup', verb='POST', path='/', params=params)