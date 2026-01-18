from yowsup.stacks import  YowStackBuilder
from .layer import EchoLayer
from yowsup.layers import YowLayerEvent
from yowsup.layers.network import YowNetworkLayer
def set_prop(self, key, val):
    self._stack.setProp(key, val)