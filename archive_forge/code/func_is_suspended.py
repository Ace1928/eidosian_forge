from typing import TYPE_CHECKING, Optional
def is_suspended(connection: 'Redis', worker: Optional['Worker']=None):
    """Checks whether a Worker is suspendeed on a given connection
    PS: pipeline returns a list of responses
    Ref: https://github.com/andymccurdy/redis-py#pipelines

    Args:
        connection (Redis): The Redis Connection
        worker (Optional[Worker], optional): The Worker. Defaults to None.
    """
    with connection.pipeline() as pipeline:
        if worker is not None:
            worker.heartbeat(pipeline=pipeline)
        pipeline.exists(WORKERS_SUSPENDED)
        return pipeline.execute()[-1]