import json
import click
from mlflow.entities import ViewType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.tracking import _get_store
from mlflow.utils.string_utils import _create_table
from mlflow.utils.time import conv_longdate_to_str
@commands.command('describe')
@RUN_ID
def describe_run(run_id):
    """
    All of run details will print to the stdout as JSON format.
    """
    store = _get_store()
    run = store.get_run(run_id)
    json_run = json.dumps(run.to_dictionary(), indent=4)
    click.echo(json_run)