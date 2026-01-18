import logging
from osc_lib.command import command
from osc_lib import utils
from monascaclient import version
@property
def mon_client(self):
    if not self._client:
        self.log.debug('Initializing mon-client')
        self._client = make_client(api_version=self.mon_version, endpoint=self.mon_url, session=self.app.client_manager.session)
    return self._client