import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_alarm_id_or_name_scenario(self):

    def _test(name):
        params = 'create --type event --name %s' % name
        result = self.aodh('alarm', params=params)
        alarm_id = self.details_multiple(result)[0]['alarm_id']
        params = 'show %s' % name
        result = self.aodh('alarm', params=params)
        self.assertEqual(alarm_id, self.details_multiple(result)[0]['alarm_id'])
        params = 'show %s' % alarm_id
        result = self.aodh('alarm', params=params)
        self.assertEqual(alarm_id, self.details_multiple(result)[0]['alarm_id'])
        params = 'update --state ok %s' % name
        result = self.aodh('alarm', params=params)
        self.assertEqual('ok', self.details_multiple(result)[0]['state'])
        params = 'update --state alarm %s' % alarm_id
        result = self.aodh('alarm', params=params)
        self.assertEqual('alarm', self.details_multiple(result)[0]['state'])
        params = 'update --name another-name %s' % name
        result = self.aodh('alarm', params=params)
        self.assertEqual('another-name', self.details_multiple(result)[0]['name'])
        params = 'update --name %s %s' % (name, alarm_id)
        result = self.aodh('alarm', params=params)
        self.assertEqual(name, self.details_multiple(result)[0]['name'])
        params = 'update --name %s %s' % (name, name)
        result = self.aodh('alarm', params=params)
        self.assertEqual(name, self.details_multiple(result)[0]['name'])
        params = 'update --state ok'
        result = self.aodh('alarm', params=params, fail_ok=True, merge_stderr=True)
        self.assertFirstLineStartsWith(result.splitlines(), 'You need to specify one of alarm ID and alarm name(--name) to update an alarm.')
        params = 'delete %s' % name
        result = self.aodh('alarm', params=params)
        self.assertEqual('', result)
        params = 'create --type event --name %s' % name
        result = self.aodh('alarm', params=params)
        alarm_id = self.details_multiple(result)[0]['alarm_id']
        params = 'delete %s' % alarm_id
        result = self.aodh('alarm', params=params)
        self.assertEqual('', result)
    _test(uuidutils.generate_uuid())
    _test('normal-alarm-name')