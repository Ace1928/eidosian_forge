import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def del_peer(self, peer, reload_config=True):
    del self.peers[peer]
    if self.is_running and reload_config:
        self.create_config()
        self.reload_config()