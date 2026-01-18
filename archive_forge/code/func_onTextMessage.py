from yowsup.layers.interface                           import YowInterfaceLayer, ProtocolEntityCallback
def onTextMessage(self, messageProtocolEntity):
    print('Echoing %s to %s' % (messageProtocolEntity.getBody(), messageProtocolEntity.getFrom(False)))