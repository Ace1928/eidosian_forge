import logging
import threading
import heapq
from botocore.compat import six
from s3transfer.compat import seekable
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import IN_MEMORY_DOWNLOAD_TAG
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import get_callbacks
from s3transfer.utils import invoke_progress_callbacks
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import FunctionContainer
from s3transfer.utils import CountCallbackInvoker
from s3transfer.utils import StreamReaderProgress
from s3transfer.utils import DeferredOpenFile
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
def queue_file_io_task(self, fileobj, data, offset):
    with self._io_submit_lock:
        writes = self._defer_queue.request_writes(offset, data)
        for write in writes:
            data = write['data']
            logger.debug('Queueing IO offset %s for fileobj: %s', write['offset'], fileobj)
            super(DownloadNonSeekableOutputManager, self).queue_file_io_task(fileobj, data, offset)