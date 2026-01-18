from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util
@property
def data_xml(self):
    """*data* element as an XML string"""
    if not self._parsed:
        self.parse()
    return to_xml(self._data)