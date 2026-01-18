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
def get_standard_attr_resource_model_map(include_resources=True, include_sub_resources=True):
    rs_map = {}
    for subclass in HasStandardAttributes.__subclasses__():
        if include_resources:
            for resource in subclass.get_api_collections():
                _resource_model_map_helper(rs_map, resource, subclass)
        if include_sub_resources:
            for sub_resource in subclass.get_api_sub_resources():
                _resource_model_map_helper(rs_map, sub_resource, subclass)
    return rs_map