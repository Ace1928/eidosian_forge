import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def delete_hsm(self, hsm_arn):
    """
        Deletes an HSM. Once complete, this operation cannot be undone
        and your key material cannot be recovered.

        :type hsm_arn: string
        :param hsm_arn: The ARN of the HSM to delete.

        """
    params = {'HsmArn': hsm_arn}
    return self.make_request(action='DeleteHsm', body=json.dumps(params))