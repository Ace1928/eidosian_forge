import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_hsm_configurations(self, hsm_configuration_identifier=None, max_records=None, marker=None):
    """
        Returns information about the specified Amazon Redshift HSM
        configuration. If no configuration ID is specified, returns
        information about all the HSM configurations owned by your AWS
        customer account.

        :type hsm_configuration_identifier: string
        :param hsm_configuration_identifier: The identifier of a specific
            Amazon Redshift HSM configuration to be described. If no identifier
            is specified, information is returned for all HSM configurations
            owned by your AWS customer account.

        :type max_records: integer
        :param max_records: The maximum number of response records to return in
            each call. If the number of remaining response records exceeds the
            specified `MaxRecords` value, a value is returned in a `marker`
            field of the response. You can retrieve the next set of records by
            retrying the command with the returned marker value.
        Default: `100`

        Constraints: minimum 20, maximum 100.

        :type marker: string
        :param marker: An optional parameter that specifies the starting point
            to return a set of response records. When the results of a
            DescribeHsmConfigurations request exceed the value specified in
            `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.

        """
    params = {}
    if hsm_configuration_identifier is not None:
        params['HsmConfigurationIdentifier'] = hsm_configuration_identifier
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeHsmConfigurations', verb='POST', path='/', params=params)