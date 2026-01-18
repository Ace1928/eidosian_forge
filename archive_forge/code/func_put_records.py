import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def put_records(self, records, stream_name, b64_encode=True):
    """
        Puts (writes) multiple data records from a producer into an
        Amazon Kinesis stream in a single call (also referred to as a
        `PutRecords` request). Use this operation to send data from a
        data producer into the Amazon Kinesis stream for real-time
        ingestion and processing. Each shard can support up to 1000
        records written per second, up to a maximum total of 1 MB data
        written per second.

        You must specify the name of the stream that captures, stores,
        and transports the data; and an array of request `Records`,
        with each record in the array requiring a partition key and
        data blob.

        The data blob can be any type of data; for example, a segment
        from a log file, geographic/location data, website clickstream
        data, and so on.

        The partition key is used by Amazon Kinesis as input to a hash
        function that maps the partition key and associated data to a
        specific shard. An MD5 hash function is used to map partition
        keys to 128-bit integer values and to map associated data
        records to shards. As a result of this hashing mechanism, all
        data records with the same partition key map to the same shard
        within the stream. For more information, see `Partition Key`_
        in the Amazon Kinesis Developer Guide .

        Each record in the `Records` array may include an optional
        parameter, `ExplicitHashKey`, which overrides the partition
        key to shard mapping. This parameter allows a data producer to
        determine explicitly the shard where the record is stored. For
        more information, see `Adding Multiple Records with
        PutRecords`_ in the Amazon Kinesis Developer Guide .

        The `PutRecords` response includes an array of response
        `Records`. Each record in the response array directly
        correlates with a record in the request array using natural
        ordering, from the top to the bottom of the request and
        response. The response `Records` array always includes the
        same number of records as the request array.

        The response `Records` array includes both successfully and
        unsuccessfully processed records. Amazon Kinesis attempts to
        process all records in each `PutRecords` request. A single
        record failure does not stop the processing of subsequent
        records.

        A successfully-processed record includes `ShardId` and
        `SequenceNumber` values. The `ShardId` parameter identifies
        the shard in the stream where the record is stored. The
        `SequenceNumber` parameter is an identifier assigned to the
        put record, unique to all records in the stream.

        An unsuccessfully-processed record includes `ErrorCode` and
        `ErrorMessage` values. `ErrorCode` reflects the type of error
        and can be one of the following values:
        `ProvisionedThroughputExceededException` or `InternalFailure`.
        `ErrorMessage` provides more detailed information about the
        `ProvisionedThroughputExceededException` exception including
        the account ID, stream name, and shard ID of the record that
        was throttled.

        Data records are accessible for only 24 hours from the time
        that they are added to an Amazon Kinesis stream.

        :type records: list
        :param records: The records associated with the request.

        :type stream_name: string
        :param stream_name: The stream name associated with the request.

        :type b64_encode: boolean
        :param b64_encode: Whether to Base64 encode `data`. Can be set to
            ``False`` if `data` is already encoded to prevent double encoding.

        """
    params = {'Records': records, 'StreamName': stream_name}
    if b64_encode:
        for i in range(len(params['Records'])):
            data = params['Records'][i]['Data']
            if not isinstance(data, six.binary_type):
                data = data.encode('utf-8')
            params['Records'][i]['Data'] = base64.b64encode(data).decode('utf-8')
    return self.make_request(action='PutRecords', body=json.dumps(params))