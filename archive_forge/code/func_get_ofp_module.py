import glob
import inspect
import os.path
from os_ken.ofproto import ofproto_protocol
def get_ofp_module(ofp_version):
    """get modules pair for the constants and parser of OF-wire of
    a given OF version.
    """
    return get_ofp_modules()[ofp_version]