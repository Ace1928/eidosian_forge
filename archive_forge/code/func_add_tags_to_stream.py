import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def add_tags_to_stream(self, stream_name, tags):
    """
        Adds or updates tags for the specified Amazon Kinesis stream.
        Each stream can have up to 10 tags.

        If tags have already been assigned to the stream,
        `AddTagsToStream` overwrites any existing tags that correspond
        to the specified tag keys.

        :type stream_name: string
        :param stream_name: The name of the stream.

        :type tags: map
        :param tags: The set of key-value pairs to use to create the tags.

        """
    params = {'StreamName': stream_name, 'Tags': tags}
    return self.make_request(action='AddTagsToStream', body=json.dumps(params))