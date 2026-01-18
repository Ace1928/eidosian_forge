import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def set_default_policy(self, peer, typ, default):
    if typ in ['in', 'out', 'import', 'export'] and default in ['reject', 'accept']:
        if 'default-policy' not in self.peers[peer]:
            self.peers[peer]['default-policy'] = {}
        self.peers[peer]['default-policy'][typ] = default
    else:
        raise Exception('wrong type or default')