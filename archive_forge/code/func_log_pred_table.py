import abc
from typing import Any, Dict, List, Optional
from tensorflow.keras.callbacks import Callback  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
def log_pred_table(self, type: str='evaluation', table_name: str='eval_data', aliases: Optional[List[str]]=None) -> None:
    """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.

        Args:
            type: (str) The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'evaluation')
            table_name: (str) The name of the table as will be displayed in the UI.
                (default is 'eval_data')
            aliases: (List[str]) List of aliases for the prediction table.
        """
    assert wandb.run is not None
    pred_artifact = wandb.Artifact(f'run_{wandb.run.id}_pred', type=type)
    pred_artifact.add(self.pred_table, table_name)
    wandb.run.log_artifact(pred_artifact, aliases=aliases or ['latest'])