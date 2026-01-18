from .notification_groups import GroupsNotificationProtocolEntity
from yowsup.structs import ProtocolTreeNode
def getCreatorJid(self, full=True):
    return self.creatorJid if full else self.creatorJid.split('@')[0]