import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def create_security_service(self, type='ldap', name=None, description=None, dns_ip=None, ou=None, server=None, domain=None, user=None, password=None, default_ad_site=None, microversion=None):
    """Creates security service.

        :param type: security service type (ldap, kerberos or active_directory)
        :param name: desired name of new security service.
        :param description: desired description of new security service.
        :param dns_ip: DNS IP address inside tenant's network.
        :param ou: security service organizational unit
        :param server: security service IP address or hostname.
        :param domain: security service domain.
        :param user: user of the new security service.
        :param password: password used by user.
        :param default_ad_site: default AD site
        """
    cmd = 'security-service-create %s ' % type
    cmd += self._combine_security_service_data(name=name, description=description, dns_ip=dns_ip, ou=ou, server=server, domain=domain, user=user, password=password, default_ad_site=default_ad_site)
    ss_raw = self.manila(cmd, microversion=microversion)
    security_service = output_parser.details(ss_raw)
    return security_service