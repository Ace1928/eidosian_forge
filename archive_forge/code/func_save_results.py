import boto
from boto.services.message import ServiceMessage
from boto.services.servicedef import ServiceDef
from boto.pyami.scriptbase import ScriptBase
from boto.utils import get_ts
import time
import os
import mimetypes
def save_results(self, results, input_message, output_message):
    output_keys = []
    for file, type in results:
        if 'OutputBucket' in input_message:
            output_bucket = input_message['OutputBucket']
        else:
            output_bucket = input_message['Bucket']
        key_name = os.path.split(file)[1]
        key = self.put_file(output_bucket, file, key_name)
        output_keys.append('%s;type=%s' % (key.name, type))
    output_message['OutputKey'] = ','.join(output_keys)