from ...state.storageprotos_pb2 import SenderKeyRecordStructure
from .senderkeystate import SenderKeyState
from ...invalidkeyidexception import InvalidKeyIdException
def isEmpty(self):
    return len(self.senderKeyStates) == 0