import requests
from wandb_gql import gql
from wandb.apis.attrs import Attrs
def invite(self, username_or_email, admin=False):
    """Invite a user to a team.

        Arguments:
            username_or_email: (str) The username or email address of the user you want to invite
            admin: (bool) Whether to make this user a team admin, defaults to False

        Returns:
            True on success, False if user was already invited or didn't exist
        """
    variables = {'entityName': self.name, 'admin': admin}
    if '@' in username_or_email:
        variables['email'] = username_or_email
    else:
        variables['username'] = username_or_email
    try:
        self._client.execute(self.CREATE_INVITE_MUTATION, variables)
    except requests.exceptions.HTTPError:
        return False
    return True