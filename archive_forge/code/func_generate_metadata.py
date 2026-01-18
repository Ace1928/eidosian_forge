import datetime
import os
import subprocess  # nosec : see comments in the code below
import uuid
from oslo_log import log
from oslo_utils import fileutils
from oslo_utils import importutils
from oslo_utils import timeutils
import saml2
from saml2 import client_base
from saml2 import md
from saml2.profile import ecp
from saml2 import saml
from saml2 import samlp
from saml2.schema import soapenv
from saml2 import sigver
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def generate_metadata(self):
    """Generate Identity Provider Metadata.

        Generate and format metadata into XML that can be exposed and
        consumed by a federated Service Provider.

        :returns: XML <EntityDescriptor> object.
        :raises keystone.exception.ValidationError: If the required
            config options aren't set.
        """
    self._ensure_required_values_present()
    entity_descriptor = self._create_entity_descriptor()
    entity_descriptor.idpsso_descriptor = self._create_idp_sso_descriptor()
    return entity_descriptor