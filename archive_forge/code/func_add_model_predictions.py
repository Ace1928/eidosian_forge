import abc
from typing import Any, Dict, List, Optional
from tensorflow.keras.callbacks import Callback  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
@abc.abstractmethod
def add_model_predictions(self, epoch: int, logs: Optional[Dict[str, float]]=None) -> None:
    """Add a prediction from a model to `pred_table`.

        Use this method to write the logic for adding model prediction for validation/
        training data to `pred_table` initialized using `init_pred_table` method.

        Example:
            ```python
            # Assuming the dataloader is not shuffling the samples.
            for idx, data in enumerate(dataloader):
                preds = model.predict(data)
                self.pred_table.add_data(
                    self.data_table_ref.data[idx][0], self.data_table_ref.data[idx][1], preds
                )
            ```
        This method is called `on_epoch_end` or equivalent hook.
        """
    raise NotImplementedError(f'{self.__class__.__name__}.add_model_predictions')