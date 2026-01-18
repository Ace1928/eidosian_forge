from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
def create_acl_entity_from_args(self, acl_manager, args):
    blank_entity = self.create_blank_acl_entity_from_uri(acl_manager, args)
    users = args.users
    if users is None:
        users = []
    else:
        users = [user for user in users if user is not None]
    entity = acl_manager.create(entity_ref=blank_entity.entity_ref, users=users, project_access=args.project_access, operation_type=args.operation_type)
    return entity