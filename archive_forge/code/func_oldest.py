import abc
from aiokafka.metrics.measurable_stat import AbstractMeasurableStat
def oldest(self, now):
    if not self._samples:
        self._samples.append(self.new_sample(now))
    oldest = self._samples[0]
    for sample in self._samples[1:]:
        if sample.last_window_ms < oldest.last_window_ms:
            oldest = sample
    return oldest