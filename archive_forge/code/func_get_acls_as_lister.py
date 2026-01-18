from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
def get_acls_as_lister(self, acl_entity):
    """Gets per operation ACL data in expected format for lister command"""
    for acl in acl_entity.operation_acls:
        setattr(acl, 'columns', acl_entity.columns)
    return acls.ACLFormatter._list_objects(acl_entity.operation_acls)