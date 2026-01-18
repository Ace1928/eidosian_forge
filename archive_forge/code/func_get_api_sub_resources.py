from oslo_utils import timeutils
import sqlalchemy as sa
from sqlalchemy import event  # noqa
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import attributes
from sqlalchemy.orm import session as se
from neutron_lib._i18n import _
from neutron_lib.db import constants as db_const
from neutron_lib.db import model_base
from neutron_lib.db import sqlalchemytypes
@classmethod
def get_api_sub_resources(cls):
    """Define the API sub-resources this object will appear under.

        This should return a list of API sub-resources that the object
        will be exposed under.

        This is used by the standard attr extensions to discover which
        sub-resources need to be extended with the standard attr fields
        (e.g. created_at/updated_at/etc).
        """
    try:
        return cls.api_sub_resources
    except AttributeError:
        return []