from tensorflow.python.framework import tensor_shape
def set_number_of_partitions(self, number_of_partitions):
    """Sets the number of partitions for the current policy.

    If the policy has been frozen then shard_dimension must match the
    existing setting.

    Args:
      number_of_partitions: The number of partitions to use in the policy.

    Raises:
      ValueError: If the policy has been frozen and shard_dimension
        differs from the frozen value.
    """
    if self._frozen:
        if self._number_of_partitions != number_of_partitions:
            raise ValueError(f"Can't set number_of_partitions to {number_of_partitions} since it has been frozen to use {self._number_of_partitions}.")
    else:
        self._number_of_partitions = number_of_partitions