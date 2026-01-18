import collections
import contextlib
import copy
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import conditions as conditions_mod
from keras_tuner.src.engine.hyperparameters import hp_types
from keras_tuner.src.engine.hyperparameters import hyperparameter as hp_module
@contextlib.contextmanager
def conditional_scope(self, parent_name, parent_values):
    """Opens a scope to create conditional HyperParameters.

        All `HyperParameter`s created under this scope will only be active when
        the parent `HyperParameter` specified by `parent_name` is equal to one
        of the values passed in `parent_values`.

        When the condition is not met, creating a `HyperParameter` under this
        scope will register the `HyperParameter`, but will return `None` rather
        than a concrete value.

        Note that any Python code under this scope will execute regardless of
        whether the condition is met.

        This feature is for the `Tuner` to collect more information of the
        search space and the current trial.  It is especially useful for model
        selection. If the parent `HyperParameter` is for model selection, the
        `HyperParameter`s in a model should only be active when the model
        selected, which can be implemented using `conditional_scope`.

        Examples:

        ```python
        def MyHyperModel(HyperModel):
            def build(self, hp):
                model = Sequential()
                model.add(Input(shape=(32, 32, 3)))
                model_type = hp.Choice("model_type", ["mlp", "cnn"])
                with hp.conditional_scope("model_type", ["mlp"]):
                    if model_type == "mlp":
                        model.add(Flatten())
                        model.add(Dense(32, activation='relu'))
                with hp.conditional_scope("model_type", ["cnn"]):
                    if model_type == "cnn":
                        model.add(Conv2D(64, 3, activation='relu'))
                        model.add(GlobalAveragePooling2D())
                model.add(Dense(10, activation='softmax'))
                return model
        ```

        Args:
            parent_name: A string, specifying the name of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.
            parent_values: A list of the values of the parent `HyperParameter`
                to use as the condition to activate the current
                `HyperParameter`.
        """
    parent_name = self._get_name(parent_name)
    if not self._exists(parent_name):
        raise ValueError(f'`HyperParameter` named: {parent_name} not defined.')
    condition = conditions_mod.Parent(parent_name, parent_values)
    self._conditions.append(condition)
    if condition.is_active(self.values):
        self.active_scopes.append(copy.deepcopy(self._conditions))
    else:
        self.inactive_scopes.append(copy.deepcopy(self._conditions))
    try:
        yield
    finally:
        self._conditions.pop()