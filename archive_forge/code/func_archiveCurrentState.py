from . import storageprotos_pb2 as storageprotos
from .sessionstate import SessionState
def archiveCurrentState(self):
    self.promoteState(SessionState())