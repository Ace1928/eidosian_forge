import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
@property
def apitools_version(self):
    return self.__apitools_version