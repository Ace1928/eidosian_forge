from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from yowsup.layers.protocol_notifications.protocolentities import NotificationProtocolEntity
from .notification_groups import GroupsNotificationProtocolEntity
def setSubjectData(self, subject, subjectOwner, subjectTimestamp):
    self.subject = subject
    self.subjectOwner = subjectOwner
    self.subjectTimestamp = int(subjectTimestamp)