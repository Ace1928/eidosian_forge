import os
import unittest
import time
from boto.swf.layer1 import Layer1
from boto.swf import exceptions as swf_exceptions
def test_list_open_workflow_executions(self):
    latest_date = time.time()
    oldest_date = time.time() - 3600
    self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date)
    self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date, tag='ig')
    self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date, workflow_id='ig')
    self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date, workflow_name='ig', workflow_version='ig')
    self.conn.list_closed_workflow_executions(self._domain, latest_date, oldest_date, reverse_order=True)