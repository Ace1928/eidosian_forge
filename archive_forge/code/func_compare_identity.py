import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
def compare_identity(boto3_creds, crt_s3_creds):
    try:
        crt_creds = crt_s3_creds()
    except botocore.exceptions.NoCredentialsError:
        return False
    is_matching_identity = boto3_creds.access_key == crt_creds.access_key_id and boto3_creds.secret_key == crt_creds.secret_access_key and (boto3_creds.token == crt_creds.session_token)
    return is_matching_identity