import math
from botocore.compat import six
from s3transfer.compat import seekable, readable
from s3transfer.futures import IN_MEMORY_UPLOAD_TAG
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import DeferredOpenFile, ChunksizeAdjuster
def get_put_object_body(self, transfer_future):
    callbacks = self._get_progress_callbacks(transfer_future)
    close_callbacks = self._get_close_callbacks(callbacks)
    fileobj = transfer_future.meta.call_args.fileobj
    body = self._wrap_data(self._initial_data + fileobj.read(), callbacks, close_callbacks)
    self._initial_data = None
    return body