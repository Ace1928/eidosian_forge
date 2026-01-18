import abc
from tensorflow.python.util.tf_export import tf_export
def create_state(self, state_manager):
    """Uses the `state_manager` to create state for the FeatureColumn.

    Args:
      state_manager: A `StateManager` to create / access resources such as
        lookup tables and variables.
    """
    pass