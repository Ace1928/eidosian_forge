from typing import Callable
import requests
from urllib3.exceptions import InsecureRequestWarning
import wandb
from wandb import env, util
from .internal import Api as InternalApi  # noqa
from .public import Api as PublicApi  # noqa
def merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings