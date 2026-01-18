from yowsup.structs import ProtocolEntity, ProtocolTreeNode
@staticmethod
def fromProtocolTreeNode(node):
    return FailureProtocolEntity(node['reason'])