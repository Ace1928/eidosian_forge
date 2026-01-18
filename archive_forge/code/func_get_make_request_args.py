import logging
import threading
from io import BytesIO
import awscrt.http
import awscrt.s3
import botocore.awsrequest
import botocore.session
from awscrt.auth import (
from awscrt.io import (
from awscrt.s3 import S3Client, S3RequestTlsMode, S3RequestType
from botocore import UNSIGNED
from botocore.compat import urlsplit
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from s3transfer.constants import MB
from s3transfer.exceptions import TransferNotDoneError
from s3transfer.futures import BaseTransferFuture, BaseTransferMeta
from s3transfer.utils import (
def get_make_request_args(self, request_type, call_args, coordinator, future, on_done_after_calls):
    request_args_handler = getattr(self, f'_get_make_request_args_{request_type}', self._default_get_make_request_args)
    return request_args_handler(request_type=request_type, call_args=call_args, coordinator=coordinator, future=future, on_done_before_calls=[], on_done_after_calls=on_done_after_calls)