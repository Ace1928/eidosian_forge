import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional
import torch
from ray.air._internal.torch_utils import (
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
@classmethod
def from_state_dict(cls, state_dict: Dict[str, Any], *, preprocessor: Optional['Preprocessor']=None) -> 'TorchCheckpoint':
    """Create a :class:`~ray.train.Checkpoint` that stores a model state dictionary.

        .. tip::

            This is the recommended method for creating
            :class:`TorchCheckpoints<TorchCheckpoint>`.

        Args:
            state_dict: The model state dictionary to store in the checkpoint.
            preprocessor: A fitted preprocessor to be applied before inference.

        Returns:
            A :class:`TorchCheckpoint` containing the specified state dictionary.

        Examples:

            .. testcode::

                import torch
                import torch.nn as nn
                from ray.train.torch import TorchCheckpoint

                # Set manual seed
                torch.manual_seed(42)

                # Function to create a NN model
                def create_model() -> nn.Module:
                    model = nn.Sequential(nn.Linear(1, 10),
                            nn.ReLU(),
                            nn.Linear(10,1))
                    return model

                # Create a TorchCheckpoint from our model's state_dict
                model = create_model()
                checkpoint = TorchCheckpoint.from_state_dict(model.state_dict())

                # Now load the model from the TorchCheckpoint by providing the
                # model architecture
                model_from_chkpt = checkpoint.get_model(create_model())

                # Assert they have the same state dict
                assert str(model.state_dict()) == str(model_from_chkpt.state_dict())
                print("worked")

            .. testoutput::
                :hide:

                ...
        """
    tempdir = tempfile.mkdtemp()
    model_path = os.path.join(tempdir, cls.MODEL_FILENAME)
    stripped_state_dict = consume_prefix_in_state_dict_if_present_not_in_place(state_dict, 'module.')
    torch.save(stripped_state_dict, model_path)
    checkpoint = cls.from_directory(tempdir)
    if preprocessor:
        checkpoint.set_preprocessor(preprocessor)
    return checkpoint