import os
import logging
from os_ken.lib import hub, alert
from os_ken.base import app_manager
from os_ken.controller import event
def start_socket_server(self):
    if not self.config.get('unixsock'):
        if self.config.get('port') is None:
            self.config['port'] = 51234
        self._start_recv_nw_sock(self.config.get('port'))
    else:
        self._start_recv()
    self.logger.info(self.config)