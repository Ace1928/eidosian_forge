import collections
import logging
import os
from typing import (
import numpy as np
import ray
from ray._private.auto_init_hook import wrap_auto_init
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.from_operators import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.plan import ExecutionPlan
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.data.dataset import Dataset, MaterializedDataset
from ray.data.datasource import (
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import Reader
from ray.data.datasource.file_based_datasource import (
from ray.data.datasource.partitioning import Partitioning
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@PublicAPI(stability='alpha')
def read_databricks_tables(*, warehouse_id: str, table: Optional[str]=None, query: Optional[str]=None, catalog: Optional[str]=None, schema: Optional[str]=None, parallelism: int=-1, ray_remote_args: Optional[Dict[str, Any]]=None) -> Dataset:
    """Read a Databricks unity catalog table or Databricks SQL execution result.

    Before calling this API, set the ``DATABRICKS_TOKEN`` environment
    variable to your Databricks warehouse access token.

    .. code-block:: console

        export DATABRICKS_TOKEN=...

    If you're not running your program on the Databricks runtime, also set the
    ``DATABRICKS_HOST`` environment variable.

    .. code-block:: console

        export DATABRICKS_HOST=adb-<workspace-id>.<random-number>.azuredatabricks.net

    .. note::

        This function is built on the
        `Databricks statement execution API <https://docs.databricks.com/api/workspace/statementexecution>`_.

    Examples:

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_databricks_tables(
                warehouse_id='...',
                catalog='catalog_1',
                schema='db_1',
                query='select id from table_1 limit 750000',
            )

    Args:
        warehouse_id: The ID of the Databricks warehouse. The query statement is
            executed on this warehouse.
        table: The name of UC table you want to read. If this argument is set,
            you can't set ``query`` argument, and the reader generates query
            of ``select * from {table_name}`` under the hood.
        query: The query you want to execute. If this argument is set,
            you can't set ``table_name`` argument.
        catalog: (Optional) The default catalog name used by the query.
        schema: (Optional) The default schema used by the query.
        parallelism: The requested parallelism of the read. Defaults to -1,
            which automatically determines the optimal parallelism for your
            configuration. You should not need to manually set this value in most cases.
            For details on how the parallelism is automatically determined and guidance
            on how to tune it, see :ref:`Tuning read parallelism
            <read_parallelism>`.
        ray_remote_args: kwargs passed to :meth:`~ray.remote` in the read tasks.

    Returns:
        A :class:`Dataset` containing the queried data.
    """
    from ray.data.datasource.databricks_uc_datasource import DatabricksUCDatasource
    from ray.util.spark.databricks_hook import get_dbutils
    from ray.util.spark.utils import get_spark_session, is_in_databricks_runtime
    token = os.environ.get('DATABRICKS_TOKEN')
    if not token:
        raise ValueError("Please set environment variable 'DATABRICKS_TOKEN' to databricks workspace access token.")
    host = os.environ.get('DATABRICKS_HOST')
    if not host:
        if is_in_databricks_runtime():
            ctx = get_dbutils().notebook.entry_point.getDbutils().notebook().getContext()
            host = ctx.tags().get('browserHostName').get()
        else:
            raise ValueError('You are not in databricks runtime, please set environment variable \'DATABRICKS_HOST\' to databricks workspace URL(e.g. "adb-<workspace-id>.<random-number>.azuredatabricks.net").')
    spark = get_spark_session()
    if not catalog:
        catalog = spark.sql('SELECT CURRENT_CATALOG()').collect()[0][0]
    if not schema:
        schema = spark.sql('SELECT CURRENT_DATABASE()').collect()[0][0]
    if query is not None and table is not None:
        raise ValueError("Only one of 'query' and 'table' arguments can be set.")
    if table:
        query = f'select * from {table}'
    if query is None:
        raise ValueError("One of 'query' and 'table_name' arguments should be set.")
    datasource = DatabricksUCDatasource(host=host, token=token, warehouse_id=warehouse_id, catalog=catalog, schema=schema, query=query)
    return read_datasource(datasource=datasource, parallelism=parallelism, ray_remote_args=ray_remote_args)