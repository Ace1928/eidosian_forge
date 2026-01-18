import re
from typing import Optional
from requests import HTTPError, Response
from ._fixes import JSONDecodeError
def append_to_message(self, additional_message: str) -> None:
    """Append additional information to the `HfHubHTTPError` initial message."""
    self.args = (self.args[0] + additional_message,) + self.args[1:]