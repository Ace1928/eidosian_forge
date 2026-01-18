from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def revrange(self, key: KeyT, from_time: Union[int, str], to_time: Union[int, str], count: Optional[int]=None, aggregation_type: Optional[str]=None, bucket_size_msec: Optional[int]=0, filter_by_ts: Optional[List[int]]=None, filter_by_min_value: Optional[int]=None, filter_by_max_value: Optional[int]=None, align: Optional[Union[int, str]]=None, latest: Optional[bool]=False, bucket_timestamp: Optional[str]=None, empty: Optional[bool]=False):
    """
        Query a range in reverse direction for a specific time-series.

        **Note**: This command is only available since RedisTimeSeries >= v1.4

        Args:

        key:
            Key name for timeseries.
        from_time:
            Start timestamp for the range query. - can be used to express the minimum possible timestamp (0).
        to_time:
            End timestamp for range query, + can be used to express the maximum possible timestamp.
        count:
            Limits the number of returned samples.
        aggregation_type:
            Optional aggregation type. Can be one of [`avg`, `sum`, `min`, `max`,
            `range`, `count`, `first`, `last`, `std.p`, `std.s`, `var.p`, `var.s`, `twa`]
        bucket_size_msec:
            Time bucket for aggregation in milliseconds.
        filter_by_ts:
            List of timestamps to filter the result by specific timestamps.
        filter_by_min_value:
            Filter result by minimum value (must mention also filter_by_max_value).
        filter_by_max_value:
            Filter result by maximum value (must mention also filter_by_min_value).
        align:
            Timestamp for alignment control for aggregation.
        latest:
            Used when a time series is a compaction, reports the compacted value of the
            latest possibly partial bucket
        bucket_timestamp:
            Controls how bucket timestamps are reported. Can be one of [`-`, `low`, `+`,
            `high`, `~`, `mid`].
        empty:
            Reports aggregations for empty buckets.

        For more information: https://redis.io/commands/ts.revrange/
        """
    params = self.__range_params(key, from_time, to_time, count, aggregation_type, bucket_size_msec, filter_by_ts, filter_by_min_value, filter_by_max_value, align, latest, bucket_timestamp, empty)
    return self.execute_command(REVRANGE_CMD, *params)