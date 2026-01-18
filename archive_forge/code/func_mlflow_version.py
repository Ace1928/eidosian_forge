from flask import request
from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics
from mlflow.version import VERSION
def mlflow_version(_: request):
    return VERSION