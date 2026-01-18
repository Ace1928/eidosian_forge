import abc
from typing import Any, Dict, List, Optional
from tensorflow.keras.callbacks import Callback  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
def log_data_table(self, name: str='val', type: str='dataset', table_name: str='val_data') -> None:
    """Log the `data_table` as W&B artifact and call `use_artifact` on it.

        This lets the evaluation table use the reference of already uploaded data
        (images, text, scalar, etc.) without re-uploading.

        Args:
            name: (str) A human-readable name for this artifact, which is how you can
                identify this artifact in the UI or reference it in use_artifact calls.
                (default is 'val')
            type: (str) The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'dataset')
            table_name: (str) The name of the table as will be displayed in the UI.
                (default is 'val_data').
        """
    data_artifact = wandb.Artifact(name, type=type)
    data_artifact.add(self.data_table, table_name)
    assert wandb.run is not None
    wandb.run.use_artifact(data_artifact)
    data_artifact.wait()
    self.data_table_ref = data_artifact.get(table_name)