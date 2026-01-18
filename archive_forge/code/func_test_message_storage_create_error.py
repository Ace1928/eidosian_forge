from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
def test_message_storage_create_error(self):
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<Error>\n  <Code>BucketAlreadyOwnedByYou</Code>\n  <Message>Your previous request to create the named bucket succeeded and you already own it.</Message>\n  <BucketName>cmsbk</BucketName>\n  <RequestId>FF8B86A32CC3FE4F</RequestId>\n  <HostId>6ENGL3DT9f0n7Tkv4qdKIs/uBNCMMA6QUFapw265WmodFDluP57esOOkecp55qhh</HostId>\n</Error>\n'
    s3ce = S3CreateError('409', 'Conflict', body=xml)
    self.assertEqual(s3ce.bucket, 'cmsbk')
    self.assertEqual(s3ce.error_code, 'BucketAlreadyOwnedByYou')
    self.assertEqual(s3ce.status, '409')
    self.assertEqual(s3ce.reason, 'Conflict')
    self.assertEqual(s3ce.error_message, 'Your previous request to create the named bucket succeeded and you already own it.')
    self.assertEqual(s3ce.error_message, s3ce.message)
    self.assertEqual(s3ce.request_id, 'FF8B86A32CC3FE4F')