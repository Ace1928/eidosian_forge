import logging
import time
import zmq
from zmq.log import handlers
from zmq.tests import BaseZMQTestCase
def test_init_iface(self):
    logger = self.logger
    ctx = self.context
    handler = handlers.PUBHandler(self.iface)
    assert handler.ctx is not ctx
    self.sockets.append(handler.socket)
    handler = handlers.PUBHandler(self.iface, self.context)
    self.sockets.append(handler.socket)
    assert handler.ctx is ctx
    handler.setLevel(logging.DEBUG)
    handler.root_topic = self.topic
    logger.addHandler(handler)
    sub = ctx.socket(zmq.SUB)
    self.sockets.append(sub)
    sub.setsockopt(zmq.SUBSCRIBE, self.topic.encode())
    sub.connect(self.iface)
    import time
    time.sleep(0.25)
    msg1 = 'message'
    logger.info(msg1)
    topic, msg2 = sub.recv_multipart()
    assert topic == b'zmq.INFO'
    assert msg2 == (msg1 + '\n').encode('utf8')
    logger.removeHandler(handler)