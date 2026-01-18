from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def end_received(self):
    if self.expecting not in ['body', 'end']:
        raise errors.SmartProtocolError('End of message received prematurely (while expecting %s)' % (self.expecting,))
    self.expecting = 'nothing'
    self.request_handler.end_received()
    if not self.request_handler.finished_reading:
        raise errors.SmartProtocolError('Complete conventional request was received, but request handler has not finished reading.')
    if not self._response_sent:
        self.responder.send_response(self.request_handler.response)