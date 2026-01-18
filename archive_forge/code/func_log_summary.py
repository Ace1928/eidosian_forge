from pathlib import Path
from types import SimpleNamespace
from typing import List, Union
from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def log_summary(model: Union[CatBoostClassifier, CatBoostRegressor], log_all_params: bool=True, save_model_checkpoint: bool=False, log_feature_importance: bool=True) -> None:
    """`log_summary` logs useful metrics about catboost model after training is done.

    Arguments:
        model: it can be CatBoostClassifier or CatBoostRegressor.
        log_all_params: (boolean) if True (default) log the model hyperparameters as W&B config.
        save_model_checkpoint: (boolean) if True saves the model upload as W&B artifacts.
        log_feature_importance: (boolean) if True (default) logs feature importance as W&B bar chart using the default setting of `get_feature_importance`.

    Using this along with `wandb_callback` will:

    - save the hyperparameters as W&B config,
    - log `best_iteration` and `best_score` as `wandb.summary`,
    - save and upload your trained model to Weights & Biases Artifacts (when `save_model_checkpoint = True`)
    - log feature importance plot.

    Example:
        ```python
        train_pool = Pool(train[features], label=train["label"], cat_features=cat_features)
        test_pool = Pool(test[features], label=test["label"], cat_features=cat_features)

        model = CatBoostRegressor(
            iterations=100,
            loss_function="Cox",
            eval_metric="Cox",
        )

        model.fit(
            train_pool,
            eval_set=test_pool,
            callbacks=[WandbCallback()],
        )

        log_summary(model)
        ```
    """
    if wandb.run is None:
        raise wandb.Error('You must call `wandb.init()` before `log_summary()`')
    if not isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
        raise wandb.Error('Model should be an instance of CatBoostClassifier or CatBoostRegressor')
    with wb_telemetry.context() as tel:
        tel.feature.catboost_log_summary = True
    params = model.get_all_params()
    if log_all_params:
        wandb.config.update(params)
    wandb.run.summary['best_iteration'] = model.get_best_iteration()
    wandb.run.summary['best_score'] = model.get_best_score()
    if save_model_checkpoint:
        aliases = ['best'] if params['use_best_model'] else ['last']
        _checkpoint_artifact(model, aliases=aliases)
    if log_feature_importance:
        _log_feature_importance(model)