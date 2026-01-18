from mock import patch, Mock
import unittest
from boto.s3.bucket import ResultSet
from boto.s3.bucketlistresultset import multipart_upload_lister
from boto.s3.bucketlistresultset import versioned_bucket_lister
def return_pages(**kwargs):
    call_args.append(kwargs)
    return pages.pop(0)