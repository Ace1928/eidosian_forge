from mlflow.protos.facet_feature_statistics_pb2 import Histogram
def generate_equal_width_histogram(quantiles, num_buckets: int, total_freq: float) -> Histogram:
    """
    Generates the equal width histogram from the input quantiles and total frequency. The
    quantiles are assumed to be ordered and corresponding to equal distant percentiles.

    Args:
        quantiles: The quantiles that capture the frequency distribution.
        num_buckets: The number of buckets in the generated histogram.
        total_freq: The total frequency (=count of rows).

    Returns:
        Equal width histogram or None if inputs are invalid.
    """
    if len(quantiles) < 2 or num_buckets <= 0 or total_freq <= 0:
        return None
    min_val = quantiles[0]
    max_val = quantiles[-1]
    histogram = Histogram()
    histogram.type = Histogram.HistogramType.STANDARD
    if min_val == max_val:
        half_buckets = num_buckets // 2
        bucket_left = min_val - half_buckets
        for i in range(num_buckets):
            if i == half_buckets:
                histogram.buckets.append(Histogram.Bucket(low_value=bucket_left, high_value=bucket_left, sample_count=total_freq))
            else:
                histogram.buckets.append(Histogram.Bucket(low_value=bucket_left, high_value=bucket_left + 1, sample_count=0))
                bucket_left += 1
    else:
        bucket_width = (max_val - min_val) / num_buckets
        for i in range(num_buckets):
            bucket_left = min_val + i * bucket_width
            bucket_right = bucket_left + bucket_width
            histogram.buckets.append(_generate_equal_width_histogram_internal(bucket_left=bucket_left, bucket_right=bucket_right, quantiles=quantiles, total_freq=total_freq))
    return histogram