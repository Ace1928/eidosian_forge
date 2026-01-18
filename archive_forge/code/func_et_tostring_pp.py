import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def et_tostring_pp(tree):
    try:
        return ET.tostring(tree, pretty_print=True)
    except TypeError:
        return ET.tostring(tree)