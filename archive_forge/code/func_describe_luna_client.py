import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def describe_luna_client(self, client_arn=None, certificate_fingerprint=None):
    """
        Retrieves information about an HSM client.

        :type client_arn: string
        :param client_arn: The ARN of the client.

        :type certificate_fingerprint: string
        :param certificate_fingerprint: The certificate fingerprint.

        """
    params = {}
    if client_arn is not None:
        params['ClientArn'] = client_arn
    if certificate_fingerprint is not None:
        params['CertificateFingerprint'] = certificate_fingerprint
    return self.make_request(action='DescribeLunaClient', body=json.dumps(params))