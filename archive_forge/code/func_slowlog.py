from aiokeydb.v1.exceptions import ResponseError, DataError
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.execution_plan import ExecutionPlan
from aiokeydb.v1.commands.graph.query_result import AsyncQueryResult, QueryResult
def slowlog(self):
    """
        Get a list containing up to 10 of the slowest queries issued
        against the given graph ID.
        For more information see `GRAPH.SLOWLOG <https://redis.io/commands/graph.slowlog>`_. # noqa

        Each item in the list has the following structure:
        1. A unix timestamp at which the log entry was processed.
        2. The issued command.
        3. The issued query.
        4. The amount of time needed for its execution, in milliseconds.
        """
    return self.execute_command(SLOWLOG_CMD, self.name)