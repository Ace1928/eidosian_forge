import copy
import os
import traceback
import warnings
from keras_tuner.src import backend
from keras_tuner.src import config as config_module
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.distribute import utils as dist_utils
from keras_tuner.src.engine import hypermodel as hm_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner_utils
def get_best_hyperparameters(self, num_trials=1):
    """Returns the best hyperparameters, as determined by the objective.

        This method can be used to reinstantiate the (untrained) best model
        found during the search process.

        Example:

        ```python
        best_hp = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hp)
        ```

        Args:
            num_trials: Optional number of `HyperParameters` objects to return.

        Returns:
            List of `HyperParameter` objects sorted from the best to the worst.
        """
    return [t.hyperparameters for t in self.oracle.get_best_trials(num_trials)]