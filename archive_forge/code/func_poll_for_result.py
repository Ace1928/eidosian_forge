import collections
import contextlib
import logging
import multiprocessing
import threading
import signal
from copy import deepcopy
import botocore.session
from botocore.config import Config
from s3transfer.constants import MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.constants import PROCESS_USER_AGENT
from s3transfer.compat import MAXINT
from s3transfer.compat import BaseManager
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import BaseTransferFuture
from s3transfer.futures import BaseTransferMeta
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import OSUtils
from s3transfer.utils import CallArgs
def poll_for_result(self, transfer_id):
    """Poll for the result of a transfer

        :param transfer_id: Unique identifier for the transfer
        :return: If the transfer succeeded, it will return the result. If the
            transfer failed, it will raise the exception associated to the
            failure.
        """
    self._transfer_states[transfer_id].wait_till_done()
    exception = self._transfer_states[transfer_id].exception
    if exception:
        raise exception
    return None