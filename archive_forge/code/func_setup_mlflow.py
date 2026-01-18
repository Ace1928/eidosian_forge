import logging
from types import ModuleType
from typing import Dict, Optional, Union
import ray
from ray.air import session
from ray.air._internal.mlflow import _MLflowLoggerUtil
from ray.air._internal import usage as air_usage
from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL
from ray.tune.experiment import Trial
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
def setup_mlflow(config: Optional[Dict]=None, tracking_uri: Optional[str]=None, registry_uri: Optional[str]=None, experiment_id: Optional[str]=None, experiment_name: Optional[str]=None, tracking_token: Optional[str]=None, artifact_location: Optional[str]=None, run_name: Optional[str]=None, create_experiment_if_not_exists: bool=False, tags: Optional[Dict]=None, rank_zero_only: bool=True) -> Union[ModuleType, _NoopModule]:
    """Set up a MLflow session.

    This function can be used to initialize an MLflow session in a
    (distributed) training or tuning run. The session will be created on the trainable.

    By default, the MLflow experiment ID is the Ray trial ID and the
    MLlflow experiment name is the Ray trial name. These settings can be overwritten by
    passing the respective keyword arguments.

    The ``config`` dict is automatically logged as the run parameters (excluding the
    mlflow settings).

    In distributed training with Ray Train, only the zero-rank worker will initialize
    mlflow. All other workers will return a noop client, so that logging is not
    duplicated in a distributed run. This can be disabled by passing
    ``rank_zero_only=False``, which will then initialize mlflow in every training
    worker.

    This function will return the ``mlflow`` module or a noop module for
    non-rank zero workers ``if rank_zero_only=True``. By using
    ``mlflow = setup_mlflow(config)`` you can ensure that only the rank zero worker
    calls the mlflow API.

    Args:
        config: Configuration dict to be logged to mlflow as parameters.
        tracking_uri: The tracking URI for MLflow tracking. If using
            Tune in a multi-node setting, make sure to use a remote server for
            tracking.
        registry_uri: The registry URI for the MLflow model registry.
        experiment_id: The id of an already created MLflow experiment.
            All logs from all trials in ``tune.Tuner()`` will be reported to this
            experiment. If this is not provided or the experiment with this
            id does not exist, you must provide an``experiment_name``. This
            parameter takes precedence over ``experiment_name``.
        experiment_name: The name of an already existing MLflow
            experiment. All logs from all trials in ``tune.Tuner()`` will be
            reported to this experiment. If this is not provided, you must
            provide a valid ``experiment_id``.
        tracking_token: A token to use for HTTP authentication when
            logging to a remote tracking server. This is useful when you
            want to log to a Databricks server, for example. This value will
            be used to set the MLFLOW_TRACKING_TOKEN environment variable on
            all the remote training processes.
        artifact_location: The location to store run artifacts.
            If not provided, MLFlow picks an appropriate default.
            Ignored if experiment already exists.
        run_name: Name of the new MLflow run that will be created.
            If not set, will default to the ``experiment_name``.
        create_experiment_if_not_exists: Whether to create an
            experiment with the provided name if it does not already
            exist. Defaults to False.
        tags: Tags to set for the new run.
        rank_zero_only: If True, will return an initialized session only for the
            rank 0 worker in distributed training. If False, will initialize a
            session for all workers. Defaults to True.

    Example:

        Per default, you can just call ``setup_mlflow`` and continue to use
        MLflow like you would normally do:

        .. code-block:: python

            from ray.air.integrations.mlflow import setup_mlflow

            def training_loop(config):
                mlflow = setup_mlflow(config)
                # ...
                mlflow.log_metric(key="loss", val=0.123, step=0)

        In distributed data parallel training, you can utilize the return value of
        ``setup_mlflow``. This will make sure it is only invoked on the first worker
        in distributed training runs.

        .. code-block:: python

            from ray.air.integrations.mlflow import setup_mlflow

            def training_loop(config):
                mlflow = setup_mlflow(config)
                # ...
                mlflow.log_metric(key="loss", val=0.123, step=0)


        You can also use MlFlow's autologging feature if using a training
        framework like Pytorch Lightning, XGBoost, etc. More information can be
        found here
        (https://mlflow.org/docs/latest/tracking.html#automatic-logging).

        .. code-block:: python

            from ray.air.integrations.mlflow import setup_mlflow

            def train_fn(config):
                mlflow = setup_mlflow(config)
                mlflow.autolog()
                xgboost_results = xgb.train(config, ...)

    """
    if not mlflow:
        raise RuntimeError('mlflow was not found - please install with `pip install mlflow`')
    try:
        _session = session._get_session(warn=False)
        if _session and rank_zero_only and (session.get_world_rank() != 0):
            return _NoopModule()
        default_trial_id = session.get_trial_id()
        default_trial_name = session.get_trial_name()
    except RuntimeError:
        default_trial_id = None
        default_trial_name = None
    _config = config.copy() if config else {}
    experiment_id = experiment_id or default_trial_id
    experiment_name = experiment_name or default_trial_name
    mlflow_util = _MLflowLoggerUtil()
    mlflow_util.setup_mlflow(tracking_uri=tracking_uri, registry_uri=registry_uri, experiment_id=experiment_id, experiment_name=experiment_name, tracking_token=tracking_token, artifact_location=artifact_location, create_experiment_if_not_exists=create_experiment_if_not_exists)
    mlflow_util.start_run(run_name=run_name or experiment_name, tags=tags, set_active=True)
    mlflow_util.log_params(_config)
    air_usage.tag_setup_mlflow()
    return mlflow_util._mlflow