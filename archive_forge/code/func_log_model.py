import json
import os
from typing import Any, Dict, List, Optional, Union
import wandb
import wandb.data_types as data_types
from wandb.data_types import _SavedModel
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
def log_model(model_obj: Any, name: str='model', aliases: Optional[Union[str, List[str]]]=None, description: Optional[str]=None, metadata: Optional[dict]=None, project: Optional[str]=None, scope_project: Optional[bool]=None, **kwargs: Dict[str, Any]) -> '_SavedModel':
    """Log a model object to enable model-centric workflows in the UI.

    Supported frameworks include PyTorch, Keras, Tensorflow, Scikit-learn, etc. Under
    the hood, we create a model artifact, bind it to the run that produced this model,
    associate it with the latest metrics logged with `wandb.log(...)` and more.

    Args:
        model_obj: any model object created with the following ML frameworks: PyTorch,
            Keras, Tensorflow, Scikit-learn. name: `str` - name of the model artifact
            that will be created to house this model_obj.
        aliases: `str, List[str]` - optional alias(es) that will be applied on this
            model and allow for unique identification. The alias "latest" will always be
            applied to the latest version of a model.
        description: `str` - text description/notes about the model - will be visible in
            the Model Card UI.
        metadata: `Dict` - model-specific metadata goes here - will be visible the UI.
        project: `str` - project under which to place this artifact.
        scope_project: `bool` - If true, name of this model artifact will not be
            suffixed by `-{run_id}`.

    Returns:
        _SavedModel instance

    Example:
        ```python
        import torch.nn as nn
        import torch.nn.functional as F


        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(10, 10)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                return x


        model = Net()
        sm = log_model(model, "my-simple-model", aliases=["best"])
        ```

    """
    model = data_types._SavedModel.init(model_obj, **kwargs)
    _ = _log_artifact_version(name=name, type='model', entries={'index': model}, aliases=aliases, description=description, metadata=metadata, project=project, scope_project=scope_project, job_type='log_model')
    return model