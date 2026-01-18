import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def delete_hsm_client_certificate(self, hsm_client_certificate_identifier):
    """
        Deletes the specified HSM client certificate.

        :type hsm_client_certificate_identifier: string
        :param hsm_client_certificate_identifier: The identifier of the HSM
            client certificate to be deleted.

        """
    params = {'HsmClientCertificateIdentifier': hsm_client_certificate_identifier}
    return self._make_request(action='DeleteHsmClientCertificate', verb='POST', path='/', params=params)