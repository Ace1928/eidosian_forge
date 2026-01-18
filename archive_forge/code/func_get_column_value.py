import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def get_column_value(manager, table, record, column):
    """
    Example : To get datapath_id from Bridge table
    get_column_value('Bridge', <bridge name>, 'datapath_id').strip('"')
    """
    row = row_by_name(manager, record, table)
    value = getattr(row, column, '')
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return str(value)