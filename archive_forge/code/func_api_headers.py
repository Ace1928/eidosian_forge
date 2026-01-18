import datetime
import sys
import time
import urllib
import platform
from typing import Any, Callable, cast, Dict, List, Optional
import json.decoder as jd
import requests
import cirq_ionq
from cirq_ionq import ionq_exceptions
from cirq import __version__ as cirq_version
def api_headers(self, api_key: str):
    """API Headers needed to make calls to the REST API.

        Args:
            api_key: The key used for authenticating against the IonQ API.

        Returns:
            dict[str, str]: A dict of :class:`requests.Request` headers.
        """
    return {'Authorization': f'apiKey {api_key}', 'Content-Type': 'application/json', 'User-Agent': self._user_agent()}