import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def create_virtual_mfa_device(self, path, device_name):
    """
        Creates a new virtual MFA device for the AWS account.

        After creating the virtual MFA, use enable-mfa-device to
        attach the MFA device to an IAM user.

        :type path: string
        :param path: The path for the virtual MFA device.

        :type device_name: string
        :param device_name: The name of the virtual MFA device.
            Used with path to uniquely identify a virtual MFA device.

        """
    params = {'Path': path, 'VirtualMFADeviceName': device_name}
    return self.get_response('CreateVirtualMFADevice', params)