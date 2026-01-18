from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import metrics_tracking
def better_than(self, a, b):
    """Whether the first objective value is better than the second.

        Args:
            a: A float, an objective value to compare.
            b: A float, another objective value to compare.

        Returns:
            Boolean, whether the first objective value is better than the
            second.
        """
    return a > b and self.direction == 'max' or (a < b and self.direction == 'min')