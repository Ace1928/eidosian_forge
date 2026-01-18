from ...utils import split_and_load
from .... import autograd
def evaluate_batch(self, estimator, val_batch, batch_axis=0):
    """Evaluate the estimator model on a batch of validation data.

        Parameters
        ----------
        estimator : Estimator
            Reference to the estimator
        val_batch : tuple
            Data and label of a batch from the validation data loader.
        batch_axis : int, default 0
            Batch axis to split the validation data into devices.
        """
    data, label = self._get_data_and_label(val_batch, estimator.context, batch_axis)
    pred = [estimator.val_net(x) for x in data]
    loss = [estimator.val_loss(y_hat, y) for y_hat, y in zip(pred, label)]
    return (data, label, pred, loss)