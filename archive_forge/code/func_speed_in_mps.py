from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message_meta import MessageMetaAttributes
from .message_media import MediaMessageProtocolEntity
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_location import LocationAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
@speed_in_mps.setter
def speed_in_mps(self, value):
    self.media_specific_attributes.speed_in_mps = value