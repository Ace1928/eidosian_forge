import os
import click
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import _get_store, fluent
from mlflow.utils.data_utils import is_uri
from mlflow.utils.string_utils import _create_table
@commands.command('csv')
@EXPERIMENT_ID
@click.option('--filename', '-o', type=click.STRING)
def generate_csv_with_runs(experiment_id, filename):
    """
    Generate CSV with all runs for an experiment
    """
    runs = fluent.search_runs(experiment_ids=experiment_id)
    if filename:
        runs.to_csv(filename, index=False)
        click.echo(f'Experiment with ID {experiment_id} has been exported as a CSV to file: {filename}.')
    else:
        click.echo(runs.to_csv(index=False))