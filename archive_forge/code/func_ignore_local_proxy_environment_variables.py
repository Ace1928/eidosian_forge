import typing
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.proxy import Proxy
def ignore_local_proxy_environment_variables(self) -> None:
    """By calling this you will ignore HTTP_PROXY and HTTPS_PROXY from
        being picked up and used."""
    self._ignore_local_proxy = True