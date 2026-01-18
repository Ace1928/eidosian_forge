import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
@mock.patch.object(loginsight.LogInsightDriver, '_append_results')
@mock.patch.object(loginsight.LogInsightDriver, '_parse_results')
def test_get_report(self, parse_results, append_results):
    start_trace = self._create_start_trace()
    start_trace['project'] = self._project
    start_trace['service'] = self._service
    stop_trace = self._create_stop_trace()
    stop_trace['project'] = self._project
    stop_trace['service'] = self._service
    resp = {'events': [{'text': 'OSProfiler trace', 'fields': [{'name': 'trace', 'content': json.dumps(start_trace)}]}, {'text': 'OSProfiler trace', 'fields': [{'name': 'trace', 'content': json.dumps(stop_trace)}]}]}
    self._client.query_events = mock.Mock(return_value=resp)
    self._driver.get_report(self.BASE_ID)
    self._client.query_events.assert_called_once_with({'base_id': self.BASE_ID})
    append_results.assert_has_calls([mock.call(start_trace['trace_id'], start_trace['parent_id'], start_trace['name'], start_trace['project'], start_trace['service'], start_trace['info']['host'], start_trace['timestamp'], start_trace), mock.call(stop_trace['trace_id'], stop_trace['parent_id'], stop_trace['name'], stop_trace['project'], stop_trace['service'], stop_trace['info']['host'], stop_trace['timestamp'], stop_trace)])
    parse_results.assert_called_once_with()