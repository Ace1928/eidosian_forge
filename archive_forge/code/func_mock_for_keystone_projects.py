import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def mock_for_keystone_projects(self, project=None, v3=True, list_get=False, id_get=False, project_list=None, project_count=None):
    if project:
        assert not (project_list or project_count)
    elif project_list:
        assert not (project or project_count)
    elif project_count:
        assert not (project or project_list)
    else:
        raise Exception('Must specify a project, project_list, or project_count')
    assert list_get or id_get
    base_url_append = 'v3' if v3 else None
    if project:
        project_list = [project]
    elif project_count:
        project_list = [self._get_project_data(v3=v3) for c in range(0, project_count)]
    uri_mock_list = []
    if list_get:
        uri_mock_list.append(dict(method='GET', uri=self.get_mock_url(service_type='identity', interface='admin', resource='projects', base_url_append=base_url_append), status_code=200, json={'projects': [p.json_response['project'] for p in project_list]}))
    if id_get:
        for p in project_list:
            uri_mock_list.append(dict(method='GET', uri=self.get_mock_url(service_type='identity', interface='admin', resource='projects', append=[p.project_id], base_url_append=base_url_append), status_code=200, json=p.json_response))
    self.__do_register_uris(uri_mock_list)
    return project_list