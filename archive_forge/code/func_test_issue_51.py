from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_issue_51(self):
    yaml = YAML()
    yaml.indent(sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.round_trip("\n        role::startup::author::rsyslog_inputs:\n          imfile:\n            - ruleset: 'AEM-slinglog'\n              File: '/opt/aem/author/crx-quickstart/logs/error.log'\n              startmsg.regex: '^[-+T.:[:digit:]]*'\n              tag: 'error'\n            - ruleset: 'AEM-slinglog'\n              File: '/opt/aem/author/crx-quickstart/logs/stdout.log'\n              startmsg.regex: '^[-+T.:[:digit:]]*'\n              tag: 'stdout'\n        ")