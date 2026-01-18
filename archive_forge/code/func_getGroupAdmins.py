from .notification_groups import GroupsNotificationProtocolEntity
from yowsup.structs import ProtocolTreeNode
def getGroupAdmins(self, full=True):
    out = []
    for jid, _type in self.participants.items():
        if _type == self.__class__.TYPE_PARTICIPANT_ADMIN:
            out.append(jid if full else jid.split('@')[0])
    return out