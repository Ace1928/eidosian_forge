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
def contextinfo_to_proto(self, contextinfo_attributes):
    cxt_info = ContextInfo()
    if contextinfo_attributes.stanza_id is not None:
        cxt_info.stanza_id = contextinfo_attributes.stanza_id
    if contextinfo_attributes.participant is not None:
        cxt_info.participant = contextinfo_attributes.participant
    if contextinfo_attributes.quoted_message:
        cxt_info.quoted_message.MergeFrom(self.message_to_proto(contextinfo_attributes.quoted_message))
    if contextinfo_attributes.remote_jid is not None:
        cxt_info.remote_jid = contextinfo_attributes.remote_jid
    if contextinfo_attributes.mentioned_jid is not None and len(contextinfo_attributes.mentioned_jid):
        cxt_info.mentioned_jid[:] = contextinfo_attributes.mentioned_jid
    if contextinfo_attributes.edit_version is not None:
        cxt_info.edit_version = contextinfo_attributes.edit_version
    if contextinfo_attributes.revoke_message is not None:
        cxt_info.revoke_message = contextinfo_attributes.revoke_message
    return cxt_info