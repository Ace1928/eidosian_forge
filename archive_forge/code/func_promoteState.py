from . import storageprotos_pb2 as storageprotos
from .sessionstate import SessionState
def promoteState(self, promotedState):
    self.previousStates.insert(0, self.sessionState)
    self.sessionState = promotedState
    if len(self.previousStates) > self.__class__.ARCHIVED_STATES_MAX_LENGTH:
        self.previousStates.pop()