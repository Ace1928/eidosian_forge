import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_list_cap(self, line):
    """list_cap <peer>
        """

    def f(p, args):
        for i in p.netconf.server_capabilities:
            print(i)
    self._request(line, f)