import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_raw_get_config(self, line):
    """raw_get_config <peer> <source>
        """

    def f(p, args):
        try:
            source = args[0]
        except:
            print('argument error')
            return
        result = p.raw_get_config(source)
        tree = ET.fromstring(result)
        validate(tree)
        print(et_tostring_pp(tree))
    self._request(line, f)