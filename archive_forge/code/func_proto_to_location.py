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
def proto_to_location(self, proto):
    return LocationAttributes(proto.degrees_latitude if proto.HasField('degrees_latitude') else None, proto.degrees_longitude if proto.HasField('degrees_longitude') else None, proto.name if proto.HasField('name') else None, proto.address if proto.HasField('address') else None, proto.url if proto.HasField('url') else None, proto.duration if proto.HasField('duration') else None, proto.accuracy_in_meters if proto.HasField('accuracy_in_meters') else None, proto.speed_in_mps if proto.HasField('speed_in_mps') else None, proto.degrees_clockwise_from_magnetic_north if proto.HasField('degrees_clockwise_from_magnetic_north') else None, proto.axolotl_sender_key_distribution_message if proto.HasField('axolotl_sender_key_distribution_message') else None, proto.jpeg_thumbnail if proto.HasField('jpeg_thumbnail') else None)