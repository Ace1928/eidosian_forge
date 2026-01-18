from yowsup.layers.protocol_messages.proto.e2e_pb2 import Message
from yowsup.layers.protocol_messages.proto.protocol_pb2 import MessageKey
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_image import ImageAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_downloadablemedia \
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_media import MediaAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_context_info import ContextInfoAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
from yowsup.layers.protocol_messages.proto.e2e_pb2 import ContextInfo
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_extendedtext import ExtendedTextAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_document import DocumentAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_contact import ContactAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_location import LocationAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_video import VideoAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_audio import AudioAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_sticker import StickerAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_sender_key_distribution_message import \
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_protocol import ProtocolAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_protocol import MessageKeyAttributes
def message_to_proto(self, message_attributes):
    message = Message()
    if message_attributes.conversation:
        message.conversation = message_attributes.conversation
    if message_attributes.image:
        message.image_message.MergeFrom(self.image_to_proto(message_attributes.image))
    if message_attributes.contact:
        message.contact_message.MergeFrom(self.contact_to_proto(message_attributes.contact))
    if message_attributes.location:
        message.location_message.MergeFrom(self.location_to_proto(message_attributes.location))
    if message_attributes.extended_text:
        message.extended_text_message.MergeFrom(self.extendedtext_to_proto(message_attributes.extended_text))
    if message_attributes.document:
        message.document_message.MergeFrom(self.document_to_proto(message_attributes.document))
    if message_attributes.audio:
        message.audio_message.MergeFrom(self.audio_to_proto(message_attributes.audio))
    if message_attributes.video:
        message.video_message.MergeFrom(self.video_to_proto(message_attributes.video))
    if message_attributes.sticker:
        message.sticker_message.MergeFrom(self.sticker_to_proto(message_attributes.sticker))
    if message_attributes.sender_key_distribution_message:
        message.sender_key_distribution_message.MergeFrom(self.sender_key_distribution_message_to_proto(message_attributes.sender_key_distribution_message))
    if message_attributes.protocol:
        message.protocol_message.MergeFrom(self.protocol_to_proto(message_attributes.protocol))
    return message