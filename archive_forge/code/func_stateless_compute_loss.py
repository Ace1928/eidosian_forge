import platform
import warnings
from keras.src import backend
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import optimizers
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.saving import serialization_lib
from keras.src.trainers.compile_utils import CompileLoss
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
from keras.src.utils import tree
def stateless_compute_loss(self, trainable_variables, non_trainable_variables, metrics_variables, x=None, y=None, y_pred=None, sample_weight=None):
    var_mapping = list(zip(self.trainable_variables, trainable_variables))
    var_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables))
    var_mapping.extend(zip(self.metrics_variables, metrics_variables))
    with backend.StatelessScope(state_mapping=var_mapping) as scope:
        loss = self.compute_loss(x, y, y_pred, sample_weight=sample_weight)
    non_trainable_variables = []
    for v in self.non_trainable_variables:
        new_v = scope.get_current_value(v)
        non_trainable_variables.append(new_v)
    metrics_variables = []
    for v in self.metrics_variables:
        new_v = scope.get_current_value(v)
        metrics_variables.append(new_v)
    return (loss, (trainable_variables, non_trainable_variables, metrics_variables))