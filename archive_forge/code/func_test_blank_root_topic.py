import logging
import time
import zmq
from zmq.log import handlers
from zmq.tests import BaseZMQTestCase
def test_blank_root_topic(self):
    logger, handler, sub_everything = self.connect_handler()
    sub_everything.setsockopt(zmq.SUBSCRIBE, b'')
    handler.socket.bind(self.iface)
    sub_only_info = sub_everything.context.socket(zmq.SUB)
    self.sockets.append(sub_only_info)
    sub_only_info.connect(self.iface)
    sub_only_info.setsockopt(zmq.SUBSCRIBE, b'INFO')
    handler.setRootTopic(b'')
    msg_debug = 'debug_message'
    logger.debug(msg_debug)
    self.assertRaisesErrno(zmq.EAGAIN, sub_only_info.recv, zmq.NOBLOCK)
    topic, msg_debug_response = sub_everything.recv_multipart()
    assert topic == b'DEBUG'
    msg_info = 'info_message'
    logger.info(msg_info)
    topic, msg_info_response_everything = sub_everything.recv_multipart()
    assert topic == b'INFO'
    topic, msg_info_response_onlyinfo = sub_only_info.recv_multipart()
    assert topic == b'INFO'
    assert msg_info_response_everything == msg_info_response_onlyinfo
    logger.removeHandler(handler)