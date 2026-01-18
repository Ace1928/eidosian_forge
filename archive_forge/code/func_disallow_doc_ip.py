import time
from boto.compat import json
def disallow_doc_ip(self, ip):
    """
        Remove the provided ip address or CIDR block from the list of
        allowable address for the document service.

        :type ip: string
        :param ip: An IP address or CIDR block you wish to grant access
            to.
        """
    arn = self.domain.doc_service_arn
    self._disallow_ip(arn, ip)