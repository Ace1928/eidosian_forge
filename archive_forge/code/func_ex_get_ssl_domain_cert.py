from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_ssl_domain_cert(self, cert_id):
    """
        Function gets the cert by id. Use this if only if the id
        is already known

        :param cert_id: The id of the specific cert
        :type cert_id: ``str``
        :returns: :class: `NttCisdomaincertificate
        """
    result = self.connection.request_with_orgId_api_2(action='networkDomainVip/sslDomainCertificate/%s' % cert_id, method='GET').object
    return self._to_cert(result)