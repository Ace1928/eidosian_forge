import zmq
from zmq.backend import Frame as FrameBase
from .attrsettr import AttributeSetter
@routing_id.setter
def routing_id(self, routing_id):
    _draft((4, 2), 'CLIENT-SERVER')
    self.set('routing_id', routing_id)