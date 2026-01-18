import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
def get_crt_s3_client(client, config):
    global CRT_S3_CLIENT
    global BOTOCORE_CRT_SERIALIZER
    with CLIENT_CREATION_LOCK:
        if CRT_S3_CLIENT is None:
            serializer, s3_client = _initialize_crt_transfer_primatives(client, config)
            BOTOCORE_CRT_SERIALIZER = serializer
            CRT_S3_CLIENT = s3_client
    return CRT_S3_CLIENT