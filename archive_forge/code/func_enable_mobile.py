import typing
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.proxy import Proxy
def enable_mobile(self, android_package: typing.Optional[str]=None, android_activity: typing.Optional[str]=None, device_serial: typing.Optional[str]=None) -> None:
    """Enables mobile browser use for browsers that support it.

        :Args:
            android_activity: The name of the android package to start
        """
    if not android_package:
        raise AttributeError('android_package must be passed in')
    self.mobile_options = {'androidPackage': android_package}
    if android_activity:
        self.mobile_options['androidActivity'] = android_activity
    if device_serial:
        self.mobile_options['androidDeviceSerial'] = device_serial