from keras.src.layers.layer import Layer
from keras.src.metrics.metric import Metric
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving import saving_lib
def map_container_variables(container, store, visited_trackables):
    if isinstance(container, dict):
        container = list(container.values())
    for trackable in container:
        if saving_lib._is_keras_trackable(trackable):
            map_trackable_variables(trackable, store, visited_trackables=visited_trackables)