import os
from tensorboard.plugins.hparams import api_pb2
def run_tag_from_session_and_metric(session_name, metric_name):
    """Returns a (run,tag) tuple storing the evaluations of the specified
    metric.

    Args:
      session_name: str.
      metric_name: MetricName protobuffer.
    Returns: (run, tag) tuple.
    """
    assert isinstance(session_name, str)
    assert isinstance(metric_name, api_pb2.MetricName)
    run = os.path.normpath(os.path.join(session_name, metric_name.group))
    tag = metric_name.tag
    return (run, tag)