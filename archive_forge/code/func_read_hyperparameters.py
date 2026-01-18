from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
def read_hyperparameters(self, ctx=None, *, experiment_ids, filters=None, sort=None):
    """Read hyperparameter values.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_ids: A Collection[string] of IDs of the enclosing
            experiments.
          filters: A Collection[HyperparameterFilter] that constrain the
            returned session groups based on hyperparameter value.
          sort: A Sequence[HyperparameterSort] that specify how the results
            should be sorted.

        Returns:
          A Collection[HyperparameterSessionGroup] describing the groups and
          their hyperparameter values.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
    return []