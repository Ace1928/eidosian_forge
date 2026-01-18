from pyviz_comms import Comm, JupyterComm
from holoviews.element.comparison import ComparisonTestCase
def assert_ready(msg=None, metadata=None):
    self.assertEqual(metadata, {'msg_type': 'Ready', 'content': '', 'comm_id': 'Testing id'})