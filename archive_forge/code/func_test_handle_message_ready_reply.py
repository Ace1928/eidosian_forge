from pyviz_comms import Comm, JupyterComm
from holoviews.element.comparison import ComparisonTestCase
def test_handle_message_ready_reply(self):

    def assert_ready(msg=None, metadata=None):
        self.assertEqual(metadata, {'msg_type': 'Ready', 'content': ''})
    comm = Comm(id='Test')
    comm.send = assert_ready
    comm._handle_msg({})