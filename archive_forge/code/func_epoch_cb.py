import time
def epoch_cb(self):
    """Callback function after each epoch. Now it records each epoch time
        and append it to epoch dataframe.
        """
    metrics = {}
    metrics['elapsed'] = self.elapsed()
    now = datetime.datetime.now()
    metrics['epoch_time'] = now - self.last_epoch_time
    self.append_metrics(metrics, 'epoch')
    self.last_epoch_time = now