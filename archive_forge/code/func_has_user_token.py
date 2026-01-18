from keystoneauth1.identity import base as base_identity
@property
def has_user_token(self):
    """Did this authentication request contained a user auth token."""
    return self.user is not None