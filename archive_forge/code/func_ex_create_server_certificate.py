import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_server_certificate(self, body: str=None, chain: str=None, name: str=None, path: str=None, private_key: str=None, dry_run: bool=False):
    """
        Creates a server certificate and its matching private key.
        These elements can be used with other services (for example,
        to configure SSL termination on load balancers).
        You can also specify the chain of intermediate certification
        authorities if your certificate is not directly signed by a root one.
        You can specify multiple intermediate certification authorities in
        the CertificateChain parameter. To do so, concatenate all certificates
        in the correct order (the first certificate must be the authority of
        your certificate, the second must the the authority of the
        first one, and so on).
        The private key must be a RSA key in PKCS1 form. To check this, open
        the PEM file and ensure its header reads as follows:
        BEGIN RSA PRIVATE KEY.
        [IMPORTANT]
        This private key must not be protected by a password or a passphrase.

        :param      body: The PEM-encoded X509 certificate. (required)
        :type       body: ``str``

        :param      chain: The PEM-encoded intermediate certification
        authorities.
        :type       chain: ``str``

        :param      name: A unique name for the certificate. Constraints:
        1-128 alphanumeric characters, pluses (+), equals (=), commas (,),
        periods (.), at signs (@), minuses (-), or underscores (_). (required)
        :type       name: ``str``

        :param      path: The path to the server certificate, set to a slash
        (/) if not specified.
        :type       path: ``str``

        :param      private_key: The PEM-encoded private key matching
        the certificate. (required)
        :type       private_key: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new server certificate
        :rtype: ``dict``
        """
    action = 'CreateServerCertificate'
    data = {'DryRun': dry_run}
    if body is not None:
        data.update({'Body': body})
    if chain is not None:
        data.update({'Chain': chain})
    if name is not None:
        data.update({'Name': name})
    if path is not None:
        data.update({'Path': path})
    if private_key is not None:
        data.update({'PrivateKey': private_key})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ServerCertificate']
    return response.json()