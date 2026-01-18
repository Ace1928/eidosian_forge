from yowsup.layers import YowProtocolLayer
from .protocolentities import ImageDownloadableMediaMessageProtocolEntity
from .protocolentities import AudioDownloadableMediaMessageProtocolEntity
from .protocolentities import VideoDownloadableMediaMessageProtocolEntity
from .protocolentities import DocumentDownloadableMediaMessageProtocolEntity
from .protocolentities import StickerDownloadableMediaMessageProtocolEntity
from .protocolentities import LocationMediaMessageProtocolEntity
from .protocolentities import ContactMediaMessageProtocolEntity
from .protocolentities import ResultRequestUploadIqProtocolEntity
from .protocolentities import MediaMessageProtocolEntity
from .protocolentities import ExtendedTextMediaMessageProtocolEntity
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity, ErrorIqProtocolEntity
import logging
def recvMessageStanza(self, node):
    if node.getAttributeValue('type') == 'media':
        mediaNode = node.getChild('proto')
        if mediaNode.getAttributeValue('mediatype') == 'image':
            entity = ImageDownloadableMediaMessageProtocolEntity.fromProtocolTreeNode(node)
            self.toUpper(entity)
        elif mediaNode.getAttributeValue('mediatype') == 'sticker':
            entity = StickerDownloadableMediaMessageProtocolEntity.fromProtocolTreeNode(node)
            self.toUpper(entity)
        elif mediaNode.getAttributeValue('mediatype') in ('audio', 'ptt'):
            entity = AudioDownloadableMediaMessageProtocolEntity.fromProtocolTreeNode(node)
            self.toUpper(entity)
        elif mediaNode.getAttributeValue('mediatype') in ('video', 'gif'):
            entity = VideoDownloadableMediaMessageProtocolEntity.fromProtocolTreeNode(node)
            self.toUpper(entity)
        elif mediaNode.getAttributeValue('mediatype') == 'location':
            entity = LocationMediaMessageProtocolEntity.fromProtocolTreeNode(node)
            self.toUpper(entity)
        elif mediaNode.getAttributeValue('mediatype') == 'contact':
            entity = ContactMediaMessageProtocolEntity.fromProtocolTreeNode(node)
            self.toUpper(entity)
        elif mediaNode.getAttributeValue('mediatype') == 'document':
            entity = DocumentDownloadableMediaMessageProtocolEntity.fromProtocolTreeNode(node)
            self.toUpper(entity)
        elif mediaNode.getAttributeValue('mediatype') == 'url':
            entity = ExtendedTextMediaMessageProtocolEntity.fromProtocolTreeNode(node)
            self.toUpper(entity)
        else:
            logger.warn('Unsupported mediatype: %s, will send receipts' % mediaNode.getAttributeValue('mediatype'))
            self.toLower(MediaMessageProtocolEntity.fromProtocolTreeNode(node).ack(True).toProtocolTreeNode())