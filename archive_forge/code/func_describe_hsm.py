import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def describe_hsm(self, hsm_arn=None, hsm_serial_number=None):
    """
        Retrieves information about an HSM. You can identify the HSM
        by its ARN or its serial number.

        :type hsm_arn: string
        :param hsm_arn: The ARN of the HSM. Either the HsmArn or the
            SerialNumber parameter must be specified.

        :type hsm_serial_number: string
        :param hsm_serial_number: The serial number of the HSM. Either the
            HsmArn or the HsmSerialNumber parameter must be specified.

        """
    params = {}
    if hsm_arn is not None:
        params['HsmArn'] = hsm_arn
    if hsm_serial_number is not None:
        params['HsmSerialNumber'] = hsm_serial_number
    return self.make_request(action='DescribeHsm', body=json.dumps(params))