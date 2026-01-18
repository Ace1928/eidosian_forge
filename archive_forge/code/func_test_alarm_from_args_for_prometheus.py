import argparse
from unittest import mock
import testtools
from aodhclient.v2 import alarm_cli
def test_alarm_from_args_for_prometheus(self):
    parser = self.cli_alarm_create.get_parser('aodh alarm create')
    test_parsed_args = parser.parse_args(['--name', 'alarm_prom', '--type', 'prometheus', '--comparison-operator', 'gt', '--threshold', '666', '--query', 'some_metric{some_label="some_value"}'])
    prom_rule = {'comparison_operator': 'gt', 'query': 'some_metric{some_label="some_value"}', 'threshold': 666.0}
    alarm_rep = self.cli_alarm_create._alarm_from_args(test_parsed_args)
    self.assertEqual(prom_rule, alarm_rep['prometheus_rule'])