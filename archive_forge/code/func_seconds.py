from .message_media_downloadable import DownloadableMediaMessageProtocolEntity
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_video import VideoAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message_meta import MessageMetaAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
@seconds.setter
def seconds(self, value):
    self.media_specific_attributes.seconds = value