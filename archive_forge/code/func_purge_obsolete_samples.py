import abc
from aiokafka.metrics.measurable_stat import AbstractMeasurableStat
def purge_obsolete_samples(self, config, now):
    """
        Timeout any windows that have expired in the absence of any events
        """
    expire_age = config.samples * config.time_window_ms
    for sample in self._samples:
        if now - sample.last_window_ms >= expire_age:
            sample.reset(now)