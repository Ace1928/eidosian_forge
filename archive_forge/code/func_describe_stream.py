import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def describe_stream(self, stream_name, limit=None, exclusive_start_shard_id=None):
    """
        Describes the specified stream.

        The information about the stream includes its current status,
        its Amazon Resource Name (ARN), and an array of shard objects.
        For each shard object, there is information about the hash key
        and sequence number ranges that the shard spans, and the IDs
        of any earlier shards that played in a role in creating the
        shard. A sequence number is the identifier associated with
        every record ingested in the Amazon Kinesis stream. The
        sequence number is assigned when a record is put into the
        stream.

        You can limit the number of returned shards using the `Limit`
        parameter. The number of shards in a stream may be too large
        to return from a single call to `DescribeStream`. You can
        detect this by using the `HasMoreShards` flag in the returned
        output. `HasMoreShards` is set to `True` when there is more
        data available.

        `DescribeStream` is a paginated operation. If there are more
        shards available, you can request them using the shard ID of
        the last shard returned. Specify this ID in the
        `ExclusiveStartShardId` parameter in a subsequent request to
        `DescribeStream`.

        `DescribeStream` has a limit of 10 transactions per second per
        account.

        :type stream_name: string
        :param stream_name: The name of the stream to describe.

        :type limit: integer
        :param limit: The maximum number of shards to return.

        :type exclusive_start_shard_id: string
        :param exclusive_start_shard_id: The shard ID of the shard to start
            with.

        """
    params = {'StreamName': stream_name}
    if limit is not None:
        params['Limit'] = limit
    if exclusive_start_shard_id is not None:
        params['ExclusiveStartShardId'] = exclusive_start_shard_id
    return self.make_request(action='DescribeStream', body=json.dumps(params))