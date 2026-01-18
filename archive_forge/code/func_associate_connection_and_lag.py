import traceback
from .retries import AWSRetry
def associate_connection_and_lag(client, connection_id, lag_id):
    try:
        AWSRetry.jittered_backoff()(client.associate_connection_with_lag)(connectionId=connection_id, lagId=lag_id)
    except botocore.exceptions.ClientError as e:
        raise DirectConnectError(msg=f'Failed to associate Direct Connect connection {connection_id} with link aggregation group {lag_id}.', last_traceback=traceback.format_exc(), exception=e)