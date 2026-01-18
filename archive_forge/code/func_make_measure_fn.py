from aiokafka.metrics.measurable import AnonMeasurable
from aiokafka.metrics.compound_stat import AbstractCompoundStat, NamedMeasurable
from .histogram import Histogram
from .sampled_stat import AbstractSampledStat
def make_measure_fn(pct):
    return lambda config, now: self.value(config, now, pct / 100.0)