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
def contact_person():
    company = md.Company(text=CONF.saml.idp_contact_company)
    given_name = md.GivenName(text=CONF.saml.idp_contact_name)
    surname = md.SurName(text=CONF.saml.idp_contact_surname)
    email = md.EmailAddress(text=CONF.saml.idp_contact_email)
    telephone = md.TelephoneNumber(text=CONF.saml.idp_contact_telephone)
    contact_type = CONF.saml.idp_contact_type
    return md.ContactPerson(company=company, given_name=given_name, sur_name=surname, email_address=email, telephone_number=telephone, contact_type=contact_type)