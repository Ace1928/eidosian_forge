import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union
import numpy as np
import tensorflow as tf
from ray.air._internal.tensorflow_utils import convert_ndarray_batch_to_tf_tensor_batch
from ray.train._internal.dl_predictor import DLPredictor
from ray.train.predictor import DataBatchType
from ray.train.tensorflow import TensorflowCheckpoint
from ray.util import log_once
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
def call_model(self, inputs: Union[tf.Tensor, Dict[str, tf.Tensor]]) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
    """Runs inference on a single batch of tensor data.

        This method is called by `TorchPredictor.predict` after converting the
        original data batch to torch tensors.

        Override this method to add custom logic for processing the model input or
        output.

        Example:

            .. testcode::

                # List outputs are not supported by default TensorflowPredictor.
                def build_model() -> tf.keras.Model:
                    input = tf.keras.layers.Input(shape=1)
                    model = tf.keras.models.Model(inputs=input, outputs=[input, input])
                    return model

                # Use a custom predictor to format model output as a dict.
                class CustomPredictor(TensorflowPredictor):
                    def call_model(self, inputs):
                        model_output = super().call_model(inputs)
                        return {
                            str(i): model_output[i] for i in range(len(model_output))
                        }

                import numpy as np
                data_batch = np.array([[0.5], [0.6], [0.7]], dtype=np.float32)

                predictor = CustomPredictor(model=build_model())
                predictions = predictor.predict(data_batch)

        Args:
            inputs: A batch of data to predict on, represented as either a single
                TensorFlow tensor or for multi-input models, a dictionary of tensors.

        Returns:
            The model outputs, either as a single tensor or a dictionary of tensors.

        """
    if self.use_gpu:
        with tf.device('GPU:0'):
            return self._model(inputs)
    else:
        return self._model(inputs)