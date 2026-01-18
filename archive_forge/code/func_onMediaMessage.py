from yowsup.layers.interface                           import YowInterfaceLayer, ProtocolEntityCallback
def onMediaMessage(self, messageProtocolEntity):
    if messageProtocolEntity.media_type == 'image':
        print('Echoing image %s to %s' % (messageProtocolEntity.url, messageProtocolEntity.getFrom(False)))
    elif messageProtocolEntity.media_type == 'location':
        print('Echoing location (%s, %s) to %s' % (messageProtocolEntity.getLatitude(), messageProtocolEntity.getLongitude(), messageProtocolEntity.getFrom(False)))
    elif messageProtocolEntity.media_type == 'contact':
        print('Echoing contact (%s, %s) to %s' % (messageProtocolEntity.getName(), messageProtocolEntity.getCardData(), messageProtocolEntity.getFrom(False)))