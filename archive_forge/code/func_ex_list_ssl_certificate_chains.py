from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
@get_params
def ex_list_ssl_certificate_chains(self, params={}):
    """
        Functions takes a named parameter that can be one or none of the
        following to filter returned items

        :param params: A sequence of comma separated keyword arguments
        and a value
            * id=
            * network_domain_id=
            * name=
            * state=
            * create_time=
            * expiry_time=
        :return: `list` of :class: `NttCissslcertficiatechain`
        """
    result = self.connection.request_with_orgId_api_2(action='networkDomainVip/sslCertificateChain', params=params, method='GET').object
    return self._to_certificate_chains(result)