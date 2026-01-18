import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def list_tags_for_stream(self, stream_name, exclusive_start_tag_key=None, limit=None):
    """
        Lists the tags for the specified Amazon Kinesis stream.

        :type stream_name: string
        :param stream_name: The name of the stream.

        :type exclusive_start_tag_key: string
        :param exclusive_start_tag_key: The key to use as the starting point
            for the list of tags. If this parameter is set, `ListTagsForStream`
            gets all tags that occur after `ExclusiveStartTagKey`.

        :type limit: integer
        :param limit: The number of tags to return. If this number is less than
            the total number of tags associated with the stream, `HasMoreTags`
            is set to `True`. To list additional tags, set
            `ExclusiveStartTagKey` to the last key in the response.

        """
    params = {'StreamName': stream_name}
    if exclusive_start_tag_key is not None:
        params['ExclusiveStartTagKey'] = exclusive_start_tag_key
    if limit is not None:
        params['Limit'] = limit
    return self.make_request(action='ListTagsForStream', body=json.dumps(params))