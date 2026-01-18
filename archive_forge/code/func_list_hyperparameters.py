from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
def list_hyperparameters(self, ctx=None, *, experiment_ids):
    """List hyperparameters metadata.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_ids: A Collection[string] of IDs of the enclosing
            experiments.

        Returns:
          A Collection[Hyperparameter] describing the hyperparameter metadata
          for the experiments.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
    return []