from yowsup.layers import YowProtocolLayer
from yowsup.layers.axolotl.protocolentities import *
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers import EventCallback
from yowsup.profile.profile import YowProfile
from yowsup.axolotl import exceptions
from yowsup.layers.axolotl.props import PROP_IDENTITY_AUTOTRUST
import logging
def receive(self, node):
    self.processIqRegistry(node)