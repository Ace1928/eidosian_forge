from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
@staticmethod
def checkValidNames(names):
    names = names if names else SetPrivacyIqProtocolEntity.NAMES
    if not type(names) is list:
        names = [names]
    for name in names:
        if not name in SetPrivacyIqProtocolEntity.NAMES:
            raise Exception("Name should be in: '" + "', '".join(SetPrivacyIqProtocolEntity.NAMES) + "' but is '" + name + "'")
    return names