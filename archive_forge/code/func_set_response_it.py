from typing import (
import asyncio
from google.api_core.exceptions import GoogleAPICallError, FailedPrecondition
from google.cloud.pubsublite.internal.wire.connection import (
from google.cloud.pubsublite.internal.wire.work_item import WorkItem
from google.cloud.pubsublite.internal.wire.permanent_failable import PermanentFailable
def set_response_it(self, response_it: AsyncIterator[Response]):
    self._response_it = response_it