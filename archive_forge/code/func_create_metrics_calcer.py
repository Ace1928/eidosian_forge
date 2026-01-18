import os
from .. import CatBoostError, CatBoost
def create_metrics_calcer(self, metrics, thread_count, eval_step=1):
    if not os.path.exists(self._model_path):
        raise CatBoostError("Model was deleted. Can't create calcer now")
    model = CatBoost()
    model.load_model(self._model_path)
    return model.create_metric_calcer(metrics, thread_count=thread_count, eval_period=eval_step)