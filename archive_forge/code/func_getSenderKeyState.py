from ...state.storageprotos_pb2 import SenderKeyRecordStructure
from .senderkeystate import SenderKeyState
from ...invalidkeyidexception import InvalidKeyIdException
def getSenderKeyState(self, keyId=None):
    if keyId is None:
        if len(self.senderKeyStates):
            return self.senderKeyStates[0]
        else:
            raise InvalidKeyIdException('No key state in record')
    else:
        for state in self.senderKeyStates:
            if state.getKeyId() == keyId:
                return state
        raise InvalidKeyIdException('No keys for: %s' % keyId)