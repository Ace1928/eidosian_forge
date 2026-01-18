from ncclient.operations import OperationError
from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
def global_operations(node):
    """Instantiate an SR OS global operation action element

    Args:
        node: A string representing the top-level action for a
            given global operation.
    Returns:
        A tuple of 'lxml.etree._Element' values.  The first value
        represents the top-level YANG action element and the second
        represents the caller supplied initial node.
    """
    parent, child = yang_action('global-operations', attrs={'xmlns': SROS_GLOBAL_OPS_NS})
    ele = sub_ele(child, node)
    return (parent, ele)