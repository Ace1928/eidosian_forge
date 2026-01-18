from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def make_cron_trigger(name, wf, workflow_input=None, pattern=None):
    return mock.Mock(name=name, pattern=pattern, workflow_input=workflow_input)