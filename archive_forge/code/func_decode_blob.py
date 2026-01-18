import boto
import re
from boto.utils import find_class
import uuid
from boto.sdb.db.key import Key
from boto.sdb.db.blob import Blob
from boto.sdb.db.property import ListProperty, MapProperty
from datetime import datetime, date, time
from boto.exception import SDBPersistenceError, S3ResponseError
from boto.compat import map, six, long_type
def decode_blob(self, value):
    if not value:
        return None
    match = re.match('^s3:\\/\\/([^\\/]*)\\/(.*)$', value)
    if match:
        s3 = self.manager.get_s3_connection()
        bucket = s3.get_bucket(match.group(1), validate=False)
        try:
            key = bucket.get_key(match.group(2))
        except S3ResponseError as e:
            if e.reason != 'Forbidden':
                raise
            return None
    else:
        return None
    if key:
        return Blob(file=key, id='s3://%s/%s' % (key.bucket.name, key.name))
    else:
        return None