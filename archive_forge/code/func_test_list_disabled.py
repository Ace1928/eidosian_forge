import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_disabled(self):
    for i in range(len(self.users)):
        user = self.users[i]
        auth = self.auths[i]
        project = self.projects[i]
        url = '/users/%s/projects?enabled=False' % user['id']
        result = self.get(url, auth=auth)
        self.assertEqual(0, len(result.json['projects']))
        project['enabled'] = False
        PROVIDERS.resource_api.update_project(project['id'], project)
        result = self.get(url, auth=auth)
        projects_result = result.json['projects']
        self.assertEqual(1, len(projects_result))
        self.assertEqual(self.projects[i]['id'], projects_result[0]['id'])