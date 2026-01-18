from .notification_groups import GroupsNotificationProtocolEntity
from yowsup.structs import ProtocolTreeNode
def getGroupSuperAdmin(self, full=True):
    for jid, _type in self.participants.items():
        if _type == self.__class__.TYPE_PARTICIPANT_SUPERADMIN:
            return jid if full else jid.split('@')[0]