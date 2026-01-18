import json
import click
from mlflow.entities import ViewType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.tracking import _get_store
from mlflow.utils.string_utils import _create_table
from mlflow.utils.time import conv_longdate_to_str
@commands.command('list')
@click.option('--experiment-id', envvar=MLFLOW_EXPERIMENT_ID.name, type=click.STRING, help='Specify the experiment ID for list of runs.', required=True)
@click.option('--view', '-v', default='active_only', help="Select view type for list experiments. Valid view types are 'active_only' (default), 'deleted_only', and 'all'.")
def list_run(experiment_id, view):
    """
    List all runs of the specified experiment in the configured tracking server.
    """
    store = _get_store()
    view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
    runs = store.search_runs([experiment_id], None, view_type)
    table = []
    for run in runs:
        run_name = run.info.run_name or ''
        table.append([conv_longdate_to_str(run.info.start_time), run_name, run.info.run_id])
    click.echo(_create_table(sorted(table, reverse=True), headers=['Date', 'Name', 'ID']))