from .storageprotos_pb2 import SignedPreKeyRecordStructure
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
def getTimestamp(self):
    return self.structure.timestamp