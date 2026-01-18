from ray.core.generated.logging_pb2 import LogBatch
def log_batch_dict_to_proto(log_json: dict) -> LogBatch:
    """Converts a dict containing a batch of logs to a LogBatch proto."""
    return LogBatch(ip=log_json.get('ip'), pid=str(log_json.get('pid')) if log_json.get('pid') else None, job_id=log_json.get('job'), is_error=bool(log_json.get('is_err')), lines=log_json.get('lines'), actor_name=log_json.get('actor_name'), task_name=log_json.get('task_name'))