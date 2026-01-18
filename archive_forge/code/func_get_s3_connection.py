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
def get_s3_connection(self):
    if not self.s3:
        self.s3 = boto.connect_s3(self.db_user, self.db_passwd)
    return self.s3