import time
@property
def mean_throughput(self):
    time_total = float(sum(self._samples))
    if not time_total:
        return 0.0
    return float(sum(self._units_processed)) / time_total