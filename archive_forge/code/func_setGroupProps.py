from .notification_groups import GroupsNotificationProtocolEntity
from yowsup.structs import ProtocolTreeNode
def setGroupProps(self, subject, participants):
    assert type(participants) is list, 'Must be a list of jids, got %s instead.' % type(participants)
    self.subject = subject
    self.participants = participants