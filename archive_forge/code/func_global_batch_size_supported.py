from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.keras import backend
from tensorflow.python.ops import variables
def global_batch_size_supported(distribution_strategy):
    return distribution_strategy.extended._global_batch_size