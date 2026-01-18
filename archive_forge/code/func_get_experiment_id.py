from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.default_experiment.abstract_context import DefaultExperimentProvider
from mlflow.utils import databricks_utils
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_SOURCE_ID, MLFLOW_EXPERIMENT_SOURCE_TYPE
def get_experiment_id(self):
    if DatabricksNotebookExperimentProvider._resolved_notebook_experiment_id:
        return DatabricksNotebookExperimentProvider._resolved_notebook_experiment_id
    source_notebook_id = databricks_utils.get_notebook_id()
    source_notebook_name = databricks_utils.get_notebook_path()
    tags = {MLFLOW_EXPERIMENT_SOURCE_ID: source_notebook_id}
    if databricks_utils.is_in_databricks_repo_notebook():
        tags[MLFLOW_EXPERIMENT_SOURCE_TYPE] = 'REPO_NOTEBOOK'
    try:
        experiment_id = MlflowClient().create_experiment(source_notebook_name, None, tags)
    except MlflowException as e:
        if e.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.INVALID_PARAMETER_VALUE):
            experiment_id = source_notebook_id
        else:
            raise e
    DatabricksNotebookExperimentProvider._resolved_notebook_experiment_id = experiment_id
    return experiment_id