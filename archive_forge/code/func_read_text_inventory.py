from .errors import BzrError
from .inventory import Inventory
def read_text_inventory(tf):
    """Return an inventory read in from tf"""
    if tf.readline() != START_MARK:
        raise BzrError('missing start mark')
    inv = Inventory()
    for l in tf:
        fields = l.split(' ')
        if fields[0] == '#':
            break
        ie = {'file_id': fields[0], 'name': unescape(fields[1]), 'kind': fields[2], 'parent_id': fields[3]}
    if l != END_MARK:
        raise BzrError('missing end mark')
    return inv