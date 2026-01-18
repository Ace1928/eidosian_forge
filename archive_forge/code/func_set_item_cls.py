import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def set_item_cls(self, cls):
    """
        While the default item class is :py:class:`boto.sdb.item.Item`, this
        default may be overridden. Use this method to change a connection's
        item class.

        :param object cls: The new class to set as this connection's item
            class. See the default item class for inspiration as to what your
            replacement should/could look like.
        """
    self.item_cls = cls