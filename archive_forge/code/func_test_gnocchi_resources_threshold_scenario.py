import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_gnocchi_resources_threshold_scenario(self):
    PROJECT_ID = uuidutils.generate_uuid()
    RESOURCE_ID = uuidutils.generate_uuid()
    req = requests.post(os.environ.get('GNOCCHI_ENDPOINT') + '/v1/resource/generic', headers={'X-Auth-Token': self.get_token()}, json={'id': RESOURCE_ID})
    self.assertEqual(201, req.status_code)
    result = self.aodh(u'alarm', params=u'create --type gnocchi_resources_threshold --name alarm_gn1 --metric cpu_util --threshold 80 --resource-id %s --resource-type generic --aggregation-method last --project-id %s' % (RESOURCE_ID, PROJECT_ID))
    alarm = self.details_multiple(result)[0]
    ALARM_ID = alarm['alarm_id']
    self.assertEqual('alarm_gn1', alarm['name'])
    self.assertEqual('cpu_util', alarm['metric'])
    self.assertEqual('80.0', alarm['threshold'])
    self.assertEqual('last', alarm['aggregation_method'])
    self.assertEqual(RESOURCE_ID, alarm['resource_id'])
    self.assertEqual('generic', alarm['resource_type'])
    result = self.aodh(u'alarm', params=u"create --type gnocchi_resources_threshold --name alarm_tc --metric cpu_util --threshold 80 --resource-id %s --resource-type generic --aggregation-method last --project-id %s --time-constraint name=cons1;start='0 11 * * *';duration=300 --time-constraint name=cons2;start='0 23 * * *';duration=600 " % (RESOURCE_ID, PROJECT_ID))
    alarm = self.details_multiple(result)[0]
    self.assertEqual('alarm_tc', alarm['name'])
    self.assertEqual('80.0', alarm['threshold'])
    self.assertIsNotNone(alarm['time_constraints'])
    self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'create --type gnocchi_resources_threshold --name alarm1 --metric cpu_util --resource-id %s --resource-type generic --aggregation-method last --project-id %s' % (RESOURCE_ID, PROJECT_ID))
    result = self.aodh('alarm', params='update %s --severity critical --threshold 90' % ALARM_ID)
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('critical', alarm_updated['severity'])
    self.assertEqual('90.0', alarm_updated['threshold'])
    result = self.aodh('alarm', params='show %s' % ALARM_ID)
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
    self.assertEqual(PROJECT_ID, alarm_show['project_id'])
    self.assertEqual('alarm_gn1', alarm_show['name'])
    self.assertEqual('cpu_util', alarm_show['metric'])
    self.assertEqual('90.0', alarm_show['threshold'])
    self.assertEqual('critical', alarm_show['severity'])
    self.assertEqual('last', alarm_show['aggregation_method'])
    self.assertEqual('generic', alarm_show['resource_type'])
    result = self.aodh('alarm', params='show --name alarm_gn1')
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
    self.assertEqual(PROJECT_ID, alarm_show['project_id'])
    self.assertEqual('alarm_gn1', alarm_show['name'])
    self.assertEqual('cpu_util', alarm_show['metric'])
    self.assertEqual('90.0', alarm_show['threshold'])
    self.assertEqual('critical', alarm_show['severity'])
    self.assertEqual('last', alarm_show['aggregation_method'])
    self.assertEqual('generic', alarm_show['resource_type'])
    self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'show %s --name alarm_gn1' % ALARM_ID)
    result = self.aodh('alarm', params='list --filter all_projects=true')
    self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
    output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
    for alarm_list in self.parser.listing(result):
        self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
        if alarm_list['alarm_id'] == ALARM_ID:
            self.assertEqual('alarm_gn1', alarm_list['name'])
    result = self.aodh('alarm', params='list --filter all_projects=true --limit 1')
    alarm_list = self.parser.listing(result)
    self.assertEqual(1, len(alarm_list))
    result = self.aodh('alarm', params='list --filter all_projects=true --sort name:asc')
    names = [r['name'] for r in self.parser.listing(result)]
    sorted_name = sorted(names)
    self.assertEqual(sorted_name, names)
    result = self.aodh(u'alarm', params=u'create --type gnocchi_resources_threshold --name alarm_th --metric cpu_util --threshold 80 --resource-id %s --resource-type generic --aggregation-method last --project-id %s ' % (RESOURCE_ID, PROJECT_ID))
    created_alarm_id = self.details_multiple(result)[0]['alarm_id']
    result = self.aodh('alarm', params='list --filter all_projects=true --sort name:asc --sort alarm_id:asc')
    alarm_list = self.parser.listing(result)
    ids_with_same_name = []
    names = []
    for alarm in alarm_list:
        names.append(['alarm_name'])
        if alarm['name'] == 'alarm_th':
            ids_with_same_name.append(alarm['alarm_id'])
    sorted_ids = sorted(ids_with_same_name)
    sorted_names = sorted(names)
    self.assertEqual(sorted_names, names)
    self.assertEqual(sorted_ids, ids_with_same_name)
    result = self.aodh('alarm', params='list --filter all_projects=true --sort name:desc --marker %s' % created_alarm_id)
    self.assertIn('alarm_tc', [r['name'] for r in self.parser.listing(result)])
    self.aodh('alarm', params='delete %s' % created_alarm_id)
    result = self.aodh('alarm', params='list --query project_id=%s' % PROJECT_ID)
    alarm_list = self.parser.listing(result)[0]
    self.assertEqual(ALARM_ID, alarm_list['alarm_id'])
    self.assertEqual('alarm_gn1', alarm_list['name'])
    result = self.aodh('alarm', params='delete %s' % ALARM_ID)
    self.assertEqual('', result)
    result = self.aodh('alarm', params='show %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
    expected = 'Alarm %s not found (HTTP 404)' % ALARM_ID
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='delete %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='list')
    self.assertNotIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])