from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def list_application_credentials(self, user_id, hints=None):
    """List application credentials for a user.

        :param str user_id: User ID
        :param dict hints: Properties to filter on

        :returns: a list of application credentials
        """
    hints = hints or driver_hints.Hints()
    app_cred_list = self.driver.list_application_credentials_for_user(user_id, hints)
    return [self._process_app_cred(app_cred) for app_cred in app_cred_list]