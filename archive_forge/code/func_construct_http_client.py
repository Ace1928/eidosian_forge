import logging
import os
import debtcollector.renames
from keystoneauth1 import access
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import importutils
import requests
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
@debtcollector.renames.renamed_kwarg('tenant_id', 'project_id', replace=True)
@debtcollector.renames.renamed_kwarg('tenant_name', 'project_name', replace=True)
def construct_http_client(username=None, user_id=None, project_name=None, project_id=None, password=None, auth_url=None, token=None, region_name=None, timeout=None, endpoint_url=None, insecure=False, endpoint_type='public', log_credentials=None, auth_strategy='keystone', ca_cert=None, cert=None, service_type='network', session=None, global_request_id=None, **kwargs):
    if session:
        kwargs.setdefault('user_agent', USER_AGENT)
        kwargs.setdefault('interface', endpoint_type)
        return SessionClient(session=session, service_type=service_type, region_name=region_name, global_request_id=global_request_id, **kwargs)
    else:
        return HTTPClient(username=username, password=password, project_id=project_id, project_name=project_name, user_id=user_id, auth_url=auth_url, token=token, endpoint_url=endpoint_url, insecure=insecure, timeout=timeout, region_name=region_name, endpoint_type=endpoint_type, service_type=service_type, ca_cert=ca_cert, cert=cert, log_credentials=log_credentials, auth_strategy=auth_strategy, global_request_id=global_request_id)