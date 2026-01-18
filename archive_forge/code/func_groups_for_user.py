from django import db
from django.contrib import auth
def groups_for_user(environ, username):
    """
    Authorize a user based on groups
    """
    db.reset_queries()
    try:
        try:
            user = UserModel._default_manager.get_by_natural_key(username)
        except UserModel.DoesNotExist:
            return []
        if not user.is_active:
            return []
        return [group.name.encode() for group in user.groups.all()]
    finally:
        db.close_old_connections()